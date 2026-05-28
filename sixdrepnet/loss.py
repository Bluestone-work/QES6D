import torch.nn as nn
import torch
import torch.nn.functional as F


def _compute_euler_angles_from_rotation_matrices(rotation_matrices):
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = torch.zeros_like(z)

    out_euler = torch.zeros(rotation_matrices.shape[0], 3, device=R.device, dtype=R.dtype)
    singular_f = singular.to(dtype=R.dtype)
    out_euler[:, 0] = x * (1.0 - singular_f) + xs * singular_f
    out_euler[:, 1] = y * (1.0 - singular_f) + ys * singular_f
    out_euler[:, 2] = z * (1.0 - singular_f) + zs * singular_f
    return out_euler


def _wrap_angle_diff_deg(pred_deg, gt_deg):
    return (pred_deg - gt_deg + 180.0) % 360.0 - 180.0
#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)


class RobustEulerAxisLoss(nn.Module):
    """
    Euler-angle robust loss with:
      1) axis weighting (yaw/pitch/roll)
      2) robust base loss (L1/Huber/Adaptive-Huber)
      3) angle-bin reweighting by |gt angle|

    Notes:
      - Euler order follows utils.compute_euler_angles_from_rotation_matrices:
        [pitch, yaw, roll].
      - axis_weights are provided in [yaw, pitch, roll] order for readability.
    """
    def __init__(
        self,
        axis_weights=(1.0, 1.0, 1.0),
        base_loss='huber',
        huber_delta_deg=4.0,
        adaptive_gamma=1.5,
        bin_edges_deg=(30.0, 60.0),
        bin_weights=(1.0, 1.3, 1.7),
        enable_angle_bin_weight=True,
        reduction='mean'
    ):
        super().__init__()
        self.base_loss = str(base_loss).lower().strip()
        self.huber_delta_deg = float(huber_delta_deg)
        self.adaptive_gamma = float(adaptive_gamma)
        self.enable_angle_bin_weight = bool(enable_angle_bin_weight)
        self.reduction = reduction

        axis_weights = torch.tensor(axis_weights, dtype=torch.float32)
        if axis_weights.numel() != 3:
            raise ValueError(f"axis_weights must have 3 values [yaw,pitch,roll], got {axis_weights.tolist()}")
        self.register_buffer('axis_weights_ypr', axis_weights)

        edges = torch.tensor(bin_edges_deg, dtype=torch.float32).flatten()
        if edges.numel() == 0:
            edges = torch.tensor([30.0, 60.0], dtype=torch.float32)
        edges, _ = torch.sort(edges)
        self.register_buffer('bin_edges_deg', edges)

        bin_weights = torch.tensor(bin_weights, dtype=torch.float32).flatten()
        expected = int(edges.numel() + 1)
        if bin_weights.numel() != expected:
            raise ValueError(
                f"bin_weights length must be len(bin_edges_deg)+1={expected}, got {bin_weights.numel()}"
            )
        self.register_buffer('bin_weights', bin_weights)

        if self.base_loss not in ('l1', 'huber', 'adaptive_huber'):
            raise ValueError(f"Unsupported base_loss: {self.base_loss}")

    def _base_loss(self, err_deg):
        abs_err = torch.abs(err_deg)
        if self.base_loss == 'l1':
            return abs_err
        if self.base_loss == 'huber':
            return F.smooth_l1_loss(
                err_deg,
                torch.zeros_like(err_deg),
                reduction='none',
                beta=max(self.huber_delta_deg, 1e-6)
            )

        huber = F.smooth_l1_loss(
            err_deg,
            torch.zeros_like(err_deg),
            reduction='none',
            beta=max(self.huber_delta_deg, 1e-6)
        )
        scale = torch.clamp(abs_err / 90.0, min=1e-6)
        focus = torch.pow(scale, max(self.adaptive_gamma - 1.0, 0.0))
        return huber * focus

    def _angle_bin_weight(self, gt_deg_abs):
        if not self.enable_angle_bin_weight:
            return torch.ones_like(gt_deg_abs)

        idx = torch.bucketize(gt_deg_abs, self.bin_edges_deg)
        return self.bin_weights[idx]

    def forward(self, R_pred, R_gt):
        pred_euler = _compute_euler_angles_from_rotation_matrices(R_pred) * (180.0 / torch.pi)
        gt_euler = _compute_euler_angles_from_rotation_matrices(R_gt) * (180.0 / torch.pi)

        pred_pitch = pred_euler[:, 0]
        pred_yaw = pred_euler[:, 1]
        pred_roll = pred_euler[:, 2]
        gt_pitch = gt_euler[:, 0]
        gt_yaw = gt_euler[:, 1]
        gt_roll = gt_euler[:, 2]

        err_yaw = _wrap_angle_diff_deg(pred_yaw, gt_yaw)
        err_pitch = _wrap_angle_diff_deg(pred_pitch, gt_pitch)
        err_roll = _wrap_angle_diff_deg(pred_roll, gt_roll)

        err_mat = torch.stack([err_yaw, err_pitch, err_roll], dim=1)
        base = self._base_loss(err_mat)

        gt_abs_mat = torch.stack([torch.abs(gt_yaw), torch.abs(gt_pitch), torch.abs(gt_roll)], dim=1)
        bin_w = self._angle_bin_weight(gt_abs_mat)

        axis_w = self.axis_weights_ypr.to(device=base.device, dtype=base.dtype).view(1, 3)
        weighted = base * bin_w * axis_w
        per_sample = weighted.sum(dim=1)

        if self.reduction == 'mean':
            return per_sample.mean()
        if self.reduction == 'sum':
            return per_sample.sum()
        return per_sample

class GeodesicPlusAxisLoss(nn.Module):
    def __init__(self, eps=1e-4, lambda_axis=0.2, lambda_ortho=0.0, reduction="mean", axis_abs=False):
        super().__init__()
        self.eps = eps
        self.lambda_axis = lambda_axis
        self.lambda_ortho = lambda_ortho
        self.reduction = reduction
        self.axis_abs = axis_abs

    def geodesic(self, R_pred, R_gt):
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos = (tr - 1.0) / 2.0
        eps = self.eps
        cos = cos.clamp(-1.0 + eps, 1.0 - eps)
        return torch.acos(cos)  # (B,)

    def axis_align(self, R_pred, R_gt):
        # normalize columns to make dot a true cosine similarity
        Rp = F.normalize(R_pred, dim=1)
        Rg = F.normalize(R_gt, dim=1)

        dots = (Rp * Rg).sum(dim=1).clamp(-1.0, 1.0)  # (B,3)
        if self.axis_abs:
            dots = dots.abs()
        return (1.0 - dots).sum(dim=1)  # (B,)

    def ortho_reg(self, R_pred):
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype).unsqueeze(0)
        RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)
        return ((RtR - I) ** 2).sum(dim=(1, 2))  # (B,)

    def forward(self, R_pred, R_gt):
        theta = self.geodesic(R_pred, R_gt)      # (B,)
        axis  = self.axis_align(R_pred, R_gt)    # (B,)
        loss  = theta + self.lambda_axis * axis

        if self.lambda_ortho > 0:
            ortho = self.ortho_reg(R_pred)
            loss = loss + self.lambda_ortho * ortho

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class GeodesicPlusOrthLoss(nn.Module):
    """
    L = L_geo + lambda_orth * L_orth
    
    L_geo: SO(3) geodesic distance
    L_orth: orthogonality constraint ||R^T R - I||_F^2
    
    强制预测的旋转矩阵保持正交性，减少数值误差累积。
    """
    def __init__(self, eps=1e-7, lambda_orth=0.1, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.lambda_orth = lambda_orth
        self.reduction = reduction

    def geodesic(self, R_pred, R_gt):
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos = (tr - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos)
        return theta

    def orthogonality(self, R_pred):
        # R^T R should be identity
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype).unsqueeze(0)  # (1,3,3)
        RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)  # (B,3,3)
        # Frobenius norm: ||RtR - I||_F^2
        diff = RtR - I
        orth_loss = (diff ** 2).sum(dim=[1, 2])  # (B,)
        return orth_loss

    def forward(self, R_pred, R_gt):
        geo_loss = self.geodesic(R_pred, R_gt)
        orth_loss = self.orthogonality(R_pred)
        
        total_loss = geo_loss + self.lambda_orth * orth_loss

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss

class AdaptiveGeodesicLoss(nn.Module):
    """
    自适应测地距离损失：对大角度误差给予更高权重（类似 Focal Loss）
    
    L = (theta / theta_max)^gamma * theta
    
    gamma > 1: 强调难样本（大角度误差）
    gamma < 1: 强调易样本
    gamma = 1: 标准测地距离
    """
    def __init__(self, eps=1e-7, gamma=2.0, theta_max=3.14159, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.theta_max = theta_max
        self.reduction = reduction

    def forward(self, R_pred, R_gt):
        R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos = (tr - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos)  # (B,)
        
        # Adaptive weight: larger errors get higher weight
        weight = (theta / self.theta_max) ** (self.gamma - 1.0)
        weighted_loss = weight * theta

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss

class EquivarianceConsistencyLoss(nn.Module):
    """
    In-plane rotation equivariance consistency loss on SO(3).

    Given a base prediction R and an augmented-image prediction R_aug,
    enforce:
        R_aug ~= Delta(T) @ R

    where Delta(T) is a known roll / in-plane rotation matrix.
    """
    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, R_pred_aug, R_target_aug):
        R_rel = torch.bmm(R_pred_aug, R_target_aug.transpose(1, 2))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos = (tr - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos)

        if self.reduction == "mean":
            return theta.mean()
        elif self.reduction == "sum":
            return theta.sum()
        else:
            return theta
