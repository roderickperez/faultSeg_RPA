DATA_DIR_NAME = "data"
STATS_DIR_NAME = "statistics"
IMAGE_DIR_NAME = "images"


# --------------------------
# Dataset controls
# --------------------------
train_count = 2                 # how many training cubes
val_count   = 1                 # how many validation cubes
mask_mode   = 1                 # 0: binary fault/no-fault, 1: normal(1) vs reverse(2)
dome_up_probability = 0.5       # fraction of cubes with 'UP' domes (0.5 => 50/50)

# --------------------------
# Cube size & padding
# --------------------------
NX = NY = NZ = 384
PAD = 128                       # cropped off all sides → final cube is 128^3

# --------------------------
# Master switches
# --------------------------
apply_deformation = True
apply_shear       = False
apply_faulting    = True
apply_noise       = True

# --------------------------
# Folding (vertical deformation)
# --------------------------
num_gaussians = 100
a0_range      = (-0.1, 0.1)
bk_range      = (0.1, 1.0)
sigma_range   = (15.0, 30.0)
sigma_micro   = (6.0, 14.0)
sigma_small   = (14.0, 28.0)
sigma_med     = (28.0, 50.0)
sigma_large   = (50.0, 90.0)
size_bucket_probs = (0.49, 0.35, 0.15, 0.01)

# Classic 2D bumps (used)
classic_num_bumps        = 10
classic_a0_range         = (-0.05, 0.05)
classic_bk_range         = (3.5, 10.0)
classic_sigma_range      = (20.0, 40.0)
classic_pos_amp_prob     = 1.0
classic_keep_within_crop = True
classic_safe_margin_frac = 0.6
classic_depth_scale      = 10.0
classic_depth_power      = 1.0

# --------------------------
# Fault admission / geometry
# --------------------------

# --------------------------
# Generate dataset (with progress bars)
# --------------------------
FAULTS_PER_CUBE_RANGE = (1, 8)  

####### or a constant number of faults
# FIXED_FAULTS = 3
# train_fault_counts = [FIXED_FAULTS]*train_count
# val_fault_counts   = [FIXED_FAULTS]*val_count
# # inclusive min/max

fault_min_cut_fraction = 0.01
fault_min_sep_z        = 6
fault_max_overlap_frac = 0.20
fault_max_proposals    = 1000 #  it’s the cap on how many random candidate faults the sampler will try per cube while attempting to admit your requested faults under spacing/overlap rules.

# Shear
e0_range = (-10.0, 10.0)
f_range  = (-0.02, 0.02)
g_range  = (-0.02, 0.02)

# Faults
max_slip_range            = (25.0, 50.0)
dip_range                 = (25, 70)
strike_range              = (0, 360)
fault_distribution_modes  = ['linear', 'gaussian']
fault_types               = ['normal', 'reverse']
fault_type_weights        = (0.5, 0.5)

# Strike sampling knob (exactly as requested)
strike_sampling_mode   = 'two_sets'      # 'two_sets' or 'random'
strike_two_set_means   = (45.0, 135.0)
strike_two_set_spread  = 12.0
strike_two_set_weights = (0.5, 0.5)

# --------------------------
# Wavelet & noise
# --------------------------
wavelet_length            = 41
wavelet_peak_freq_range   = (20.0, 35.0)
noise_type                = 'gaussian'
noise_intensity           = 0.05

# Reproducibility
np.random.seed(42)