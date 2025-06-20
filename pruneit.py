import argparse
import os
import yaml

import torch
from torch import nn
from models.common import Conv, C3, SPPF, Concat
from models.yolo import Detect
from prune_utils.pruning_elementary import prune_conv_yolo, adjust_conv_yolo, adjust_conv
from prune_utils.pruning_component import prune_m_backbone, prune_m_head, prune_c3_backbone, prune_sppf_backbone, prune_c3_head

# --- Function to load YAML ---
def load_prune_ratios(config_file_path):
    """
    Loads pruning ratios from a YAML configuration file.

    Args:
        config_file_path (str): The path to the YAML file containing the prune ratios.

    Returns:
        dict: The dictionary of pruning ratios, or an empty dictionary if loading fails.
    """
    if not os.path.exists(config_file_path):
        print(f"Error: Pruning config file '{config_file_path}' not found. Please provide a valid path.")
        return {}
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            prune_ratios_data = yaml.safe_load(file)
            
            if not isinstance(prune_ratios_data, dict):
                print(f"Warning: Expected a dictionary in '{config_file_path}', but got {type(prune_ratios_data)}. Returning empty ratios.")
                return {}
            
            # Ensure keys are integers and values are floats
            cleaned_ratios = {int(k): float(v) for k, v in prune_ratios_data.items()}
            return cleaned_ratios
            
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_file_path}': {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading '{config_file_path}': {e}")
        return {}

def prune_model(prune_ratios, model, filename):
    # compatibility with yolov5s
    save_dict = {}
    # copy everything else
    for key, item in model.items():
        save_dict[key] = item
    
    detection_model = model["model"]
    
    
    # lists containing indexes with of layers with multiple connections
    froms = []
    tos = []
    kept_mask_layers_idx = []
    for i, l in enumerate(detection_model.yaml["backbone"]+detection_model.yaml["head"]) :
        froms.append(l[0])
        kept_mask_layers_idx += l[0] if isinstance(l[0],list) else [l[0]]
        tos.append(i)
    kept_mask_layers_idx = [i for i in kept_mask_layers_idx if i!=-1]

    
    kept_mask_layers = {}
    kept_mask = None
    kept_mask_prev = kept_mask
    
    for i, l in enumerate(detection_model.model):
    
        prune_ratio = prune_ratios[i]
        
        
        # If backbone layer (backbone's bottlenecks have skip connections)
        if i <= 9:
            
            if isinstance(l, Conv):
                kept_mask = prune_conv_yolo(l,prune_ratio,kept_mask_prev)
    
            elif isinstance(l, C3):
                
                #print(i, l.m[0].add)
                kept_mask=prune_c3_backbone(l,prune_ratio,kept_mask_prev)
    
            elif isinstance(l, SPPF):
                kept_mask = prune_sppf_backbone(l,prune_ratio,kept_mask_prev)
                
            else:
                print("WHICH LAYER IS THIS:",l)
                break
                
            # If this layer connects to multiple nodes, then save its mask
            if i in kept_mask_layers_idx:
                # add key value
                kept_mask_layers[i]=kept_mask
    
    
        # If not backbone, head. Head's bottlenecks have no skip connections
        else:
            """
            Head notes:
            following code is ran when list of froms or int but other than -1. Except detect, only Concat layer has list of froms.
            The following check helps in detecting lists as well
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    
            So concatenation happens in order [previos layer corresponding to -1, other layers corresponding to non -1 indexs]
    
            we will use this same f attribute to get 'from' layers
            """
    
            # If layer is of type Concat, then concat the filter masks
            if isinstance(l, Concat):
                
                # If multiple inputs, then get their masks
                if isinstance(l.f, list):
                    kept_mask = torch.cat([kept_mask_prev if j == -1 else kept_mask_layers[j] for j in l.f])
            
            elif isinstance(l, Conv):
                kept_mask = prune_conv_yolo(l,prune_ratio,kept_mask_prev)
    
            elif isinstance(l, C3):
                #print("kept_mask_prev:", kept_mask_prev)
                #print(i, l.m[0].add)
    
                kept_mask=prune_c3_head(l,prune_ratio,kept_mask_prev)
    
            elif isinstance(l, nn.Upsample):
                continue
    
            # If Detect instance, then adjust each detecion conv layer. do not prune them.
            elif isinstance(l, Detect):
                if isinstance(l.f, list): # This will always be true since Detect head gets 3 inputs for its three detection conv layers
                    for k,(in_idx, cv) in enumerate(zip(l.f, l.m)): # m attribute of Detect contains 3 Conv layers
                        cv_adj, _ = adjust_conv(kept_mask_layers[in_idx], cv) 
                        l.m[k]=cv_adj
            else:
                print("Which layer")
            
    
        # If the layer index belongs to the layer that's connected > 1, to head layers, then store its mask
        # this doesnt store masks of previous layers, unless that layer also has its output connected to multiple layers
        if i in kept_mask_layers_idx:
                # add key value
                kept_mask_layers[i]=kept_mask
            
        kept_mask_prev = kept_mask
    
            
    print("------------------------------------------------------------------------------------------")
    print("Pruning Completed!!!!")
    
    
    save_dict["model"] = detection_model.half()
    save_dict["best_fitness"] = None
    torch.save(save_dict, filename+'_spruned.pt')
    print("Saved pruned model ",filename+'_spruned.pt')
        

# --- Main script logic ---
def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(description="Run YOLOv5 model with custom pruning ratios from a YAML file.")
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the YAML file containing prune ratios (e.g., prune_ratios.yaml)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        help='Path to the pretrained YOLOv5 model file (e.g., /kaggle/input/racket-3cls/racket-3cls-im416-datal.pt)'
    )

    # 2. Parse the arguments from the command line
    args = parser.parse_args()

    config_file_path = args.config
    yolov5_model_path = args.model
    filename = yolov5_model_path.split("/")[-1].split(".")[0]
    print(filename)

    print(f"Loading configuration from: {config_file_path}")
    print(f"Loading YOLOv5 model from: {yolov5_model_path}")

    # 3. Load prune ratios from the YAML file
    prune_ratios = load_prune_ratios(config_file_path)

    if not prune_ratios:
        print("Failed to load prune ratios or ratios are empty. Exiting.")
        return # Or handle fallback to default ratios here if desired

    print("\n--- Loaded Prune Ratios ---")
    for layer, ratio in prune_ratios.items():
        print(f"Layer {layer}: {ratio}")
    print("--------------------------\n")

    # 4. Load the pretrained YOLOv5 model
    try:
        
        # Simple torch.load for demonstration
        model = torch.load(yolov5_model_path)
        print(f"Successfully loaded YOLOv5 model.")
        
        
    except FileNotFoundError:
        print(f"Error: YOLOv5 model file '{yolov5_model_path}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the YOLOv5 model: {e}")
        return

    print("\n--- Starting main script execution ---")
    prune_model(prune_ratios, model, filename)

    
    print("using the 'prune_ratios' dictionary and the 'model' object.")
    print("\n--- Script execution finished ---")

if __name__ == "__main__":
    main()
