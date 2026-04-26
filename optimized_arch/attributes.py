import torch
from helper_func import get_rand_id

# openai/clip-vit-base-patch32
# openai/clip-vit-large-patch14


class Configuration:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_set_device = 0

    # Model
    model_name: str = '--'
    v_pretrain_weight: str = 'openai/clip-vit-large-patch14'
    t_pretrain_weight: str = 'openai/clip-vit-large-patch14'

    expID = -1
    embed_dim: int = 768 #CLS:512 or 768, patch:1024, 
    save_weights = True
   
    # Adapters
    v_adapter_id = "ybelkada/opt-350m-lora"
    t_adapter_id = "ybelkada/opt-350m-lora"
    v_use_adapter = True
    t_use_adapter = True
    use_ptrain_adapter = False


    # Training
    epochs: int = 20            # Match Fahim's baseline (was 30)
    lr = 0.00001                # Match Fahim's baseline: 1e-5
    batch_size: int = 32        # Match Fahim's baseline (was 64)
    fusion_mode: str = 'linear'  # E4: Independent linear projections (W_sat·xr + W_txt·xt)
    lang_with: str = 'sat'      # fuse text with satellite embeddings
    train_eval_per_epoch = 2
    use_mixed_precision = True
    warmup_epochs = 2

    # Augmentation (ConGeo upgrades)
    use_fov_aug = False         # DISABLE: Train set is already pre-cropped 90° FoV
    use_zero_padding = False    # DISABLE: Only applies to full panos
    use_congeo_loss = False     # V9a: OFF — proven harmful (-3.6% R@1)
    congeo_weight = 0.1         # DECREASE: Keep auxiliary loss from dominating the main InfoNCE loss
    unfreeze_backbone = False    # Disabled: all params trainable from epoch 0 (Fahim baseline)

    # Loss upgrades
    use_arcgeo_loss = False     # DISABLE: Documented cold-start collapse failure
    arcgeo_margin = 0.5236      # 30° in radians (pi/6)
    use_dwbl = False            # Keep off to cleanly isolate the ArcGeo margin gains



    # Eval
    save_vis_embed = False
    use_vis_embed = False
    gnd_embed = []
    sat_embed = []
    # gnd_embed_pretrn = torch.load("embeddings/clip_cvusa_gnd.pt")
    # sat_embed_pretrn = torch.load("embeddings/clip_cvusa_sat.pt")
    batch_no = 0
    eval_size = -1



    # Data
    data_path = '/lustre/fs1/home/at387336/LGAlign_project' # Set this to the dataset directory on your cluster
    dataset_nm = "CVUSA" #CVUSA or #CVACT or #VIGOR or #GAMa
    eval_db = "CVUSA"
    lang = 'T1' # T1, T2 or T3
    use_neg_text = False
    num_workers = 5

    # Loss
    loss_margin = 1 # TripletMarginLoss
    label_smoothing=0.5 # Contrastive Loss



    #others
    msg: str = 'unset'
    latlong_csv = ''







