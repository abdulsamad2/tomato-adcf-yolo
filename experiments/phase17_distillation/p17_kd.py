"""Phase 17 localization-focused distillation loss (training-time only; inference graph unchanged).

    L = L_box + L_cls + L_dfl          unchanged Ultralytics v8DetectionLoss (same assigner, same gains)
      + lam_ld * hyp.dfl * L_LD        box-distribution distillation on the student's foreground anchors
      + lam_at * L_AT                  P3/P4 spatial-attention feature distillation

  L_LD = mean_{a in fg} mean_{side=1..4} KL( softmax(t_{a,side}/T) || softmax(s_{a,side}/T) ) * T^2
         t, s = teacher / student box-branch logits over the reg_max=16 DFL bins; fg = the student's
         TaskAlignedAssigner foreground mask that Ultralytics already computes (ground-truth based).
  L_AT = sum_{l in {P3, P4}} mean_b || q_S,l - q_T,l ||_2^2,   q = vec(mean_c F_l^2) / ||vec(mean_c F_l^2)||_2
         channel-free, so no adapter layers and no extra trainable parameters.
  Like the detection loss, the total is multiplied by the batch size (Ultralytics convention).
  No class-logit distillation. Teacher frozen: eval mode, requires_grad False, torch.no_grad forward.
  The criterion is a plain Python object (not an nn.Module) attached AFTER the EMA copy is made, and the
  trainer strips `criterion` before every save, so no teacher tensor or KD state enters any checkpoint.
  Logged loss items stay the three detection terms, so training curves are comparable with Phase 15.
"""
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


class KDDetectionLoss(v8DetectionLoss):
    def __init__(self, student, teacher, lam_ld, lam_at, t_ld=1.0, probe=False):
        super().__init__(student)          # same construction as DetectionModel.init_criterion (no one2one head)
        self.teacher, self.lam_ld, self.lam_at, self.t_ld = teacher, float(lam_ld), float(lam_at), float(t_ld)
        self.probe = probe                 # compute (and log) KD terms even when both lambdas are 0
        self.log = {"n": 0, "ld": 0.0, "at": 0.0, "det": 0.0}
        self.batch_log, self.first_shapes = [], None

    @staticmethod
    def _att(f):
        return F.normalize(f.float().pow(2).mean(1).flatten(1), p=2, dim=1)

    def kd_terms(self, preds, batch, fg_mask):
        with torch.no_grad():
            tp = self.teacher(batch["img"])
            tp = tp[1] if isinstance(tp, tuple) else tp
        if self.first_shapes is None:
            self.first_shapes = {"student_boxes": list(preds["boxes"].shape), "teacher_boxes": list(tp["boxes"].shape),
                                 "student_feats": [list(f.shape) for f in preds["feats"]],
                                 "teacher_feats": [list(f.shape) for f in tp["feats"]],
                                 "fg_anchors_first_batch": int(fg_mask.sum())}
        s_box = preds["boxes"].permute(0, 2, 1).float()
        t_box = tp["boxes"].permute(0, 2, 1).float()
        if fg_mask.any():
            s = s_box[fg_mask].view(-1, 4, self.reg_max) / self.t_ld
            t = t_box[fg_mask].view(-1, 4, self.reg_max) / self.t_ld
            ld = (F.softmax(t, -1) * (F.log_softmax(t, -1) - F.log_softmax(s, -1))).sum(-1).mean() * self.t_ld ** 2
        else:
            ld = s_box.sum() * 0.0
        at = sum(((self._att(preds["feats"][l]) - self._att(tp["feats"][l])) ** 2).sum(1).mean() for l in (0, 1))
        return ld, at

    def loss(self, preds, batch):
        batch_size = preds["boxes"].shape[0]
        (fg_mask, *_), loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)
        if self.lam_ld == 0.0 and self.lam_at == 0.0 and not self.probe:
            return loss * batch_size, loss_detach                  # KD off: the ordinary path, untouched
        ld, at = self.kd_terms(preds, batch, fg_mask)
        self.log["n"] += 1; self.log["ld"] += float(ld.detach()); self.log["at"] += float(at.detach())
        self.log["det"] += float(loss.detach().sum())
        self.batch_log.append((float(loss.detach().sum()), float(ld.detach()), float(at.detach())))
        kd = self.lam_ld * self.hyp.dfl * ld + self.lam_at * at
        return (loss.sum() + kd) * batch_size, loss_detach


def attach(trainer, teacher_path, lam_ld, lam_at, t_ld=1.0, probe=False):
    """Install the KD criterion on the trainer's (already EMA-copied) model. Call from on_train_start."""
    from ultralytics import YOLO
    student = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    teacher = YOLO(str(teacher_path)).model.float().to(trainer.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    crit = KDDetectionLoss(student, teacher, lam_ld, lam_at, t_ld, probe)
    student.criterion = crit
    return crit, teacher
