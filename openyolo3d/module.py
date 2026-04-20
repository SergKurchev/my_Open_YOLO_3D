from .utils.gt_utils import get_3d_gt_consensus, quaternion_to_matrix

def compute_3d_iou(pred_masks, gt_masks):
    """
    Compute IoU matrix between pred_masks [M, N] and gt_masks [K, N].
    N is number of points.
    """
    pred_masks = pred_masks.float()
    gt_masks = gt_masks.float()
    intersection = torch.mm(pred_masks, gt_masks.t())
    union = pred_masks.sum(1, keepdim=True) + gt_masks.sum(1, keepdim=True).t() - intersection
    return intersection / (union + 1e-6)

def evaluate_panoptic(iou_matrix, threshold=0.5):
    """
    Compute TP, FP, FN based on IoU threshold and matching.
    """
    # Simple greedy matching
    matched_gt = set()
    tps = []
    
    # Sort by IoU
    val, idx = iou_matrix.max(1)
    for i in range(len(val)):
        if val[i] >= threshold and idx[i].item() not in matched_gt:
            tps.append(val[i].item())
            matched_gt.add(idx[i].item())
            
    tp = len(tps)
    fp = iou_matrix.shape[0] - tp
    fn = iou_matrix.shape[1] - tp
    
    sq = sum(tps) / tp if tp > 0 else 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0
    pq = sq * rq
    
    return pq, sq, rq

class OpenYolo3DModule(pl.LightningModule):
    def __init__(
        self,
        config_path: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        text_prompts: Optional[List[str]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config_path = config_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.text_prompts = text_prompts or ['ripe strawberry', 'unripe strawberry', 'half_ripe strawberry', 'peduncle']
        
        self.model = OpenYolo3D(config_path)
        self.validation_step_outputs = []

    def forward(self, scene_path: str, datatype: str = "point cloud") -> Dict:
        predictions = self.model.predict(
            path_2_scene_data=scene_path,
            depth_scale=1000.0,
            text=self.text_prompts,
            datatype=datatype
        )
        return predictions

    def training_step(self, batch, batch_idx):
        # Implementation of fine-tuning heads
        loss = torch.tensor(0.1, requires_grad=True, device=self.device)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        scene_path = batch[0] if isinstance(batch, list) else batch
        
        # 1. Run prediction
        predictions_raw = self.forward(scene_path)
        scene_name = list(predictions_raw.keys())[0]
        pred_masks, pred_classes, pred_scores = predictions_raw[scene_name]
        
        # 2. Get Ground Truth (from 2D to 3D consensus)
        # This part should be cached/precomputed in real training
        # For evaluation, we assume GT exists or compute it on the fly
        logger.info(f"Generating 3D GT for {scene_name}...")
        # (Simplified loading of views/poses for example)
        # gt_instance_ids, gt_category_ids = get_3d_gt_consensus(...)
        
        # Placeholder for real GT comparison till dataloader provides GT paths
        val_status = 1.0 # Mocked for now
        
        self.log("val_status", val_status, prog_bar=True)
        return {"scene": scene_name, "val_status": val_status}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.network_3d.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
