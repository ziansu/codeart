CONFIG=$1
CURRENT_DIR=$(pwd)

WANDB_org=your_org
WANDB_project=type-inference

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

torchrun --nproc_per_node=2 --master_port=$2 run.py $CURRENT_DIR/$CONFIG