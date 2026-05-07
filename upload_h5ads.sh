#!/bin/bash

# Configuration
RCLONE_BIN="/opt/homebrew/bin/rclone"
REMOTE_NAME="gdrive"
FOLDER_ID="1qJ6hinlGj9kgFKG5lyJmcA7Z7eXuim7c"
TARGET_DIR="KaroSpaceDW"
DESTINATION="${REMOTE_NAME},root_id='${FOLDER_ID}':${TARGET_DIR}"

# List of files to upload (updated with /Volumes/processing)
FILES=(
    "/private/tmp/tissuumaps_extract_20260505/data/files/MS_xenium_data_v5_with_images_tmap.companion.ready.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/codex3.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/coxms3.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/dbit-30258697/E13_20_fig3.companion.ready.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/dbit-30258697/E13_50_2.companion.ready.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/raw/merfish/processed_mapped/adata_coronal_mapped_scVI_GMM.h5ad"
    "/Volumes/processing2/KaroSpaceDataWrangle/raw/merfish/processed_mapped/adata_sagittal_mapped_scVI_GMM.h5ad"
    "/Volumes/processing2/erectile_dysfunction/data/adata/ED_5k_filtered_clustered_cytetype_cellcharter_metadata_subset_updated_mana.h5ad"
    "/Volumes/processing2/RRmap/data/RRmap_metadata_fixed_update.h5ad"
    "/Volumes/processing2/RRmap/data/RREAE_5k_raw_only_integration_processed_updated_annotation_compartment_cytetype.h5ad"
    "/Volumes/processing2/RRmap/data/cell_paper_mana.h5ad"
    "/Volumes/processing2/ST_BRICHOS/data/ST_BRICHOS_region_subcluster.h5ad"
    "/Volumes/processing2/ST_BRICHOS/data/ST_BRICHOS_region_subcluster.companion.ready.h5ad"
    "/Volumes/processing2/autism/autism_concatenated_filtered_sparse_315genes.companion.ready.h5ad"
    "/Volumes/processing2/autism/autism_concatenated_filtered_sparse_485genes.companion.ready.h5ad"
    "/Volumes/processing2/autism/autism_concatenated_filtered_sparse.companion.ready.h5ad"
    "/Volumes/processing2/dbit-nature/data/adata_multi_pp-mana.h5ad"
    "/Volumes/processing2/nature-dev-mouse-reanalysis/ad_all_processed_with_polygons_mana.h5ad"
    "/Volumes/processing2/BALO/baloMS-nuclei-clustered-sc.h5ad"
    "/Volumes/processing2/oligo-mtDSB/data/mtDNA_DSB_5k_clustered_annotation_MANA.h5ad"
    "/Volumes/processing/cellxgene/cellxgene_visium_merged/breast_breast_carcinoma_10.1038_s41588-021-00911-1.companion.ready.h5ad"
    "/Volumes/processing/cellxgene/cellxgene_visium_merged/hindlimb_10.1038_s41586-023-06806-x.companion.ready.h5ad"
    "/Volumes/processing/cellxgene/cellxgene_visium_merged/lung_10.1016_j.cell.2022.11.005.companion.ready.h5ad"
    "/Volumes/processing/cellxgene/cellxgene_visium_merged/outflow_tract_myocardium_nodoi.companion.ready.h5ad"
    "/Volumes/processing/cellxgene/wrangled/psc-visum.companion.ready.h5ad"
    "/Volumes/processing/GSE282203_combined.companion.ready.h5ad"
    "/Volumes/processing/RRmap_with_refined_anno_and_myeloid_lineage_and_UCell_signature_scoring_scanpy_scores.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/baloMS_indep_clust_balo_MANA_SC.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/GSE253710.processed.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/GSE284005_merfish_all.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/Human_VisiumHD_compressed_v1.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/muBrainRelease_seurat.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/Nanostring_CosMX_v1.companion.ready.h5ad"
    "/Users/chrislangseth/Downloads/talbot_xenium_tumor_annotated_updated_cellcharter.companion.ready.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/xenium_cml/atera_human-breast_cancer.companion.ready.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/processed/dev-heart/dev_heart_combined.companion.ready.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/processed/glioblastoma-xenium/glioblastoma-xenium-combined-analysis.companion.ready_with_polygons_sample_id_remapped.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/processed/GSE284089/GSM8677818.companion.ready.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/humanDevMeninges/humanDevMeninges_loom_celllevel_combined.h5ad"
    "/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling/data/visium_data/RR3-brain_ST_final_object.companion.ready.h5ad"
)

echo "Starting upload to Google Drive folder: ${FOLDER_ID}/${TARGET_DIR}"
echo "Total files to process: ${#FILES[@]}"
echo "------------------------------------------------"

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "Uploading: $(basename "$FILE")"
        $RCLONE_BIN copy "$FILE" "$DESTINATION" --progress --stats-one-line
    else
        echo "Warning: File not found, skipping: $FILE"
    fi
done

echo "------------------------------------------------"
echo "Upload process complete."
