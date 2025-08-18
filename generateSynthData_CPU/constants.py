DATA_DIR_NAME = "data"
STATS_DIR_NAME = "statistics"
IMAGE_DIR_NAME = "images"


# --------------------------
# Dataset controls
# --------------------------
train_count = 200                 # how many training cubes
val_count   = 20                 # how many validation cubes
mask_mode   = 0                 # 0: binary fault/no-fault, 1: normal(1) vs reverse(2)
dome_up_probability = 0.5       # fraction of cubes with 'UP' domes (0.5 => 50/50)

FAULTS_PER_CUBE_RANGE = (2, 4)  

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
classic_polarity         = 'up'         # 'up' = anticline (dome up), 'down' = syncline (dome down)

# --------------------------

fault_min_cut_fraction = 0.01
fault_min_sep_z        = 6
fault_max_overlap_frac = 0.20
fault_max_proposals    = 1000 #  it’s the cap on how many random candidate faults the sampler will try per cube while attempting to admit your requested faults under spacing/overlap rules.

# Shear
e0_range = (-10.0, 10.0)   # constant vertical shift
f_range  = (-0.02, 0.02)   # shear gradient along X
g_range  = (-0.02, 0.02)   # shear gradient along Y

# Fault generation parameters
fault_distribution_modes = ['linear', 'gaussian']  # slip distribution modes
fault_types            = ['normal', 'reverse']     # fault type categories
fault_type_weights     = (0.5, 0.5)                # probability for normal vs. reverse fault
strike_sampling_mode   = 'two_sets'                # 'two_sets' (bimodal strikes) or 'random'
strike_two_set_means   = (45.0, 135.0)             # means of the two strike sets (degrees)
strike_two_set_spread  = 12.0                      # spread (±) around each mean (degrees)
strike_two_set_weights = (0.5, 0.5)                # weights for each strike set
dip_range              = (45, 60)                # dip angle range (degrees from horizontal)
strike_range           = (0, 360)                  # strike angle range (degrees)
max_slip_range         = (25.0, 50.0)              # fault slip magnitude range (in voxels)
fault_min_cut_fraction = 0.01    # minimum fraction of volume that a fault plane must cut to be accepted
fault_min_sep_z        = 6       # minimum vertical separation (in voxels) between any two faults
fault_max_overlap_frac = 0.20    # maximum allowable overlap (fraction of area) between fault planes
fault_max_proposals    = 1000    # max attempts to sample fault planes for each cube

# Wavelet and noise parameters
wavelet_length          = 41
wavelet_peak_freq_range = (20.0, 35.0)   # Peak frequency range for Ricker wavelet (Hz)
noise_type              = 'gaussian'     # 'gaussian', 'uniform', 'speckle', or 'salt_pepper'
noise_intensity         = 0.05           # Noise intensity factor relative to data range

# --------------------------
# Reproducibility
# --------------------------
RNG_SEED = 42  # set in the notebook with np.random.seed(RNG_SEED)

# --------------------------
# Colors used in plots/viewers
# --------------------------
# Matplotlib RGB tuples (Stats)
normal_colour  = (0.50, 0.00, 0.50)   # purple (Normal)
inverse_colour = (0.00, 0.80, 0.00)   # green  (Reverse)


# Plotly RGBA strings (3D viewers)
RGBA_RED    = "rgba(255,0,0,0.95)"
RGBA_GREEN  = "rgba(0,204,0,0.95)"
RGBA_PURPLE = "rgba(128,0,128,0.95)"

# Tiny offsets for 3D intersections (avoid z-fighting)
EPS_X = EPS_Y = EPS_Z = 1e-3

LABEL_NORMAL = 1
LABEL_REVERSE = 2