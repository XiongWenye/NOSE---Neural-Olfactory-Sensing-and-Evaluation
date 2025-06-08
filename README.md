# NOSE
# NOSE: Neural Olfactory Sensing and Evaluation

Repository for Introduction to Machine Learning Course Project, 2025 Spring.

This project provides code for neural-based olfactory (smell) sensing and evaluation using various machine learning models.

---

## Folder Structure

### `train/`

Contains scripts for model training and experiments:

- [`classification.py`](train/classification.py): Classification tasks on olfactory data.
- [`regression.py`](train/regression.py): Regression tasks for scent-related properties.
- [`fine-tuned MolFormer.py`](train/fine-tuned%20MolFormer.py): Fine-tuning MolFormer model.
- [`finetune_multitask.py`](train/finetune_multitask.py): Multitask fine-tuning.
- [`OpenPoM.py`](train/OpenPoM.py): OpenPoM-related training.

### `utils/`

Helper functions, dataset preparation, and visualization:

- [`prepare_datasets.py`](utils/prepare_datasets.py): Dataset preparation utilities.
- [`gs_lf.py`](utils/gs_lf.py): Latent factor helper.
- [`helper_methods.py`](utils/helper_methods.py): Miscellaneous helper functions.
- [`mol_loss.py`](utils/mol_loss.py): Molecular loss calculations.
- [`util_alignment.py`](utils/util_alignment.py): Alignment utilities.
- [`visualization_helper.py`](utils/visualization_helper.py): Visualization tools.
- [`test_gs_lf.ipynb`](utils/test_gs_lf.ipynb): Notebook for testing latent factor code.

### `custom_utils/`

Custom utilities for argument parsing, data handling, encoding, and configuration:

- [`args.py`](custom_utils/args.py), [`args_finetune.py`](custom_utils/args_finetune.py): Argument parsing.
- [`data_utils.py`](custom_utils/data_utils.py): Data loading and preprocessing.
- [`pubchem_encoder.py`](custom_utils/pubchem_encoder.py): PubChem encoding utilities.
- [`hparams.yaml`](custom_utils/hparams.yaml): Hyperparameter configuration.
- [`train_pubchem_light.py`](custom_utils/train_pubchem_light.py): Training on PubChem-light.
- [`pubchem_canon_zinc_final_vocab_sorted.pth`](custom_utils/pubchem_canon_zinc_final_vocab_sorted.pth): Precomputed vocabulary (PyTorch format).
- Folders: [`rotate_attention/`](custom_utils/rotate_attention), [`tokenizer/`](custom_utils/tokenizer)

*(For the full file listing, see the [custom_utils folder here](https://github.com/XiongWenye/NOSE---Neural-Olfactory-Sensing-and-Evaluation/tree/master/custom_utils).)*

---

## How to Run

1. **Install Requirements**

   The environment for this project can be quite tricky. We encourage you follow the exact steps of IBM's MoLFormer repository to set up the environment. You can find the instructions [here](https://github.com/IBM/molformer)

   Warning: The environment includes apex, which may fail in certain CUDA versions. If you encounter issues, try using a different CUDA version or change the optimization method to Adam in the training scripts.

2. **Prepare Datasets**

   We use the curated GS-LF dataset. You can download it from [here](https://github.com/ARY2260/openpom/blob/74e964eb5b1086badcb3e3ba47df3528259d7000/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv).

   For the Keller-2016 dataset used as test set, you can download it from [here](https://github.com/pyrfume/pyrfume-data/tree/main/keller_2016). We also add an extra binarization step to the dataset.

3. **Train Models**

   Before running the training scripts, ensure you have the datasets prepared and placed in the correct directories. You will also need to download the MoLFormer_Pretrained model from [here](https://github.com/Farzaneh-Taleb/MoLFormer_N2024) Notice that the checkpoint files are vital for your fine-tuning process. Make sure to have them before you run the training scripts.

   For fine-tuning specific models:

   ```bash
   python train/finetune_multitask.py
   python train/fine-tuned\ MolFormer.py
   ```

    For training classification or regression models:
    
    ```bash
    python train/classification.py
    python train/regression.py
    ```

4. **Customize Arguments and Hyperparameters**

   - Edit YAML config in `custom_utils/hparams.yaml` for hyperparameters.
   - Use scripts in `custom_utils/args.py` or `custom_utils/args_finetune.py` for advanced argument parsing.

---