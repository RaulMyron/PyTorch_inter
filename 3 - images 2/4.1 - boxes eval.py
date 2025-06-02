import torch
from torchvision.ops import box_iou
import numpy as np

# Assume box_a, box_b, box_c, and bbox are torch.int tensors with shape [1, 4]
# Example dummy data (replace with your actual tensors)
box_a = torch.tensor([[10, 10, 50, 50]], dtype=torch.int)
box_b = torch.tensor([[20, 20, 60, 60]], dtype=torch.int)
box_c = torch.tensor([[30, 30, 70, 70]], dtype=torch.int)
bbox = torch.tensor([[15, 15, 55, 55]], dtype=torch.int)

# Stack predicted boxes into a single tensor
pred_boxes = torch.cat([box_a, box_b, box_c], dim=0)  # shape: [3, 4]
gt_box = bbox  # shape: [1, 4]

# Compute IoU between each predicted box and the ground truth box
ious = box_iou(pred_boxes, gt_box).squeeze(1)  # shape: [3]

# Find the index of the box with the highest IoU
best_idx = ious.argmax().item()
box_names = ['box_a', 'box_b', 'box_c']
print(f"Closest box to ground truth: {box_names[best_idx]} (IoU={ious[best_idx]:.4f})")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dummy image (replace with your actual image if available)
img = np.ones((80, 80, 3), dtype=np.uint8) * 255

fig, ax = plt.subplots(1)
ax.imshow(img)

# Draw predicted boxes
colors = ['r', 'g', 'b']
for i, box in enumerate(pred_boxes):
    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors[i], facecolor='none', label=box_names[i])
    ax.add_patch(rect)

# Draw ground truth box
x1, y1, x2, y2 = gt_box[0].tolist()
rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='orange', facecolor='none', label='gt_box')
ax.add_patch(rect)

# Add legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()