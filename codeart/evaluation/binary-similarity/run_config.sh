CONFIG=$1
CURRENT_DIR=$(pwd)

WANDB_org=<your_wandb_org>
WANDB_project=codeart-binsim

# Check if the CONFIG file is provided
if [ -z "$CONFIG" ]; then
    echo "Please provide a config file"
    exit 1
fi

# Check if the CONFIG file exists
if [ ! -f "$CURRENT_DIR/$CONFIG" ]; then
    echo "Config file $CURRENT_DIR/$CONFIG does not exist"
    exit 1
fi

echo "Config: $CONFIG"

WANDB_ENTITY=$WANDB_org \
WANDB_PROJECT=$WANDB_project \
torchrun --nproc_per_node=2 run.py $CURRENT_DIR/$CONFIG