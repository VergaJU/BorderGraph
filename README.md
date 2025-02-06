# BorderGraph

## Description

**BorderGraph** is a Python package designed to work with **Squidpy** for computing cell contours from a segmentation mask and constructing a **neighborhood graph** based on **cell borders** and a user-defined distance threshold.  

Unlike conventional methods that rely on **cell centroids**, **BorderGraph** allows for a more precise neighborhood graph by explicitly considering **cell borders**. This is particularly useful in cases where larger cells, such as **adipocytes**, need to be included in the analysis without requiring large distance thresholds.  

## Features
- **Contours Extraction**: Computes **cell borders** from a segmentation mask.
- **Precise Neighborhood Graph**: Establishes connections based on **border proximity** rather than centroids.
- **Distance Thresholding**: Allows fine-tuned control over neighborhood connections.
- **Squidpy Integration**: Compatible with **Squidpy** for spatial transcriptomics analysis.

## Use Case Example
When analyzing **adipocytes**, conventional methods might require **large distance thresholds** to account for their size, leading to **false connections** between smaller cells. **BorderGraph** resolves this by directly considering **cell borders**, ensuring **biologically meaningful** neighborhood graphs.

```
import BorderGraph as bg
## Compute contours of each cell in the dataset:
adata=bg.Contours.contours_per_image(adata)
# Compute neighborhood graph and update adata
results = bg.Distances.compute_all_images(adata, cutoff=cutoff)
adata = bg.Utils.update_adata(adata,key_added, results, cutoff)

```
