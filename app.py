import gradio as gr
import os
import torch
import subprocess
import json

TABLE_1_MODELS = {
    "JTrans": "PurCL/jtrans-malware-2f-100c",
    "CodeArt": "PurCL/codeart-26m",
}


TABLE_2_MODELS = {
    # "PurCL/codeart-26m-mfc-2f-100c": "CodeArtForMultipleSequenceClassification",
    # "PurCL/jtrans-mfc": "JTransForMultipleSequenceClassification",
    "2Funcs-JTrans": "PurCL/jtrans-malware-2f-100c",
    "2Funcs-CodeArt": "PurCL/codeart-26m-mfc-2f-100c",
    "3Funcs-JTrans": "PurCL/jtrans-malware-3f-100c",
    "3Funcs-CodeArt": "PurCL/codeart-26m-mfc-3f-100c",
    "4Funcs-JTrans": "PurCL/jtrans-malware-4f-100c",
    "4Funcs-CodeArt": "PurCL/codeart-26m-mfc-4f-100c",
}

TABLE_3_MODELS = {
    "PurCL/codeart-3m": "CodeArtForMaskedLMWithEdgePrediction",
    "PurCL/codeart-3m-wo_local_mask": "CodeArtForMaskedLMWithEdgePrediction",
    "PurCL/codeart-3m-wo_rel_pos_bias": "CodeArtForMaskedLMWithEdgePrediction",
    "PurCL/codeart-3m-max_trans_closure_4": "CodeArtForMaskedLMWithEdgePrediction",
    "PurCL/codeart-3m-max_trans_closure_6": "CodeArtForMaskedLMWithEdgePrediction",
    "PurCL/codeart-3m-wo_trans_closure": "RabertForMaskedLMWithEdgePrediction",
}

FIG_9_DATASET = {
    "O0": ("PurCL/marinda-type-inference-debuginfo-only-O0-shuffle"),
    "O1": ("PurCL/marinda-type-inference-debuginfo-only-O1-shuffle"),
    "O2": ("PurCL/marinda-type-inference-debuginfo-only-O2-shuffle"),
    "O3": ("PurCL/marinda-type-inference-debuginfo-only-O3-shuffle"),
}

FIG_9_MODELS = {
    "O0": "PurCL/codeart-26m-ti-O0",
    "O1": "PurCL/codeart-26m-ti-O1",
    "O2": "PurCL/codeart-26m-ti-O2",
    "O3": "PurCL/codeart-26m-ti-O3",
}

GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEIVCE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


def run_table2_eval(model_name, progress=gr.Progress()):
    progress(0, desc="Starting")
    
    cwd = os.getcwd()
    if "codeart" in model_name.lower():
        os.chdir(f"{cwd}/codeart/evaluation/malware-family-classification")

        with open("config/eval-2f-100c.json", "r") as f:
            config = json.load(f)

    elif "jtrans" in model_name.lower():
        os.chdir(f"{cwd}/codeart/evaluation-jtrans/malware-family-classification")

        with open("config/eval.json", "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Invalid model name")

    config["model_name_or_path"] = TABLE_2_MODELS[model_name]
    # "max_functions": 4
    config["max_functions"] = int(model_name.split("-")[-2][0])
    progress(0.2, desc="Starting")

    command = ["python3", "evaluate_multilabel.py"]
    for key, value in config.items():
        command += [f"--{key}", str(value)]

    output = subprocess.run(command, capture_output=True, text=True)

    os.chdir(cwd)

    return output.stdout

def run_fig9_eval(model_name, optimization_level, progress=gr.Progress()):
    assert optimization_level in model_name
    cwd = os.getcwd()
    os.chdir(f"{cwd}/codeart/evaluation/type-inference")
    progress(0.1, desc="Starting")
    with open("config/eval-O0.json", "r") as f:
        config = json.load(f)

    config["model_name_or_path"] = model_name
    config["dataset_name"] = FIG_9_DATASET[optimization_level]
    
    # delete report_to key
    config.pop("report_to", None)
    progress(0.2, desc="Starting")
    command = ["python3", "run.py"]
    for key, value in config.items():
        command += [f"--{key}", str(value)]

    output = subprocess.run(command, capture_output=True, text=True)

    os.chdir(cwd)

    return output.stdout

def run_fig8_tab1_eval(model_name, run_alias, dataset_name, pool_size, progress=gr.Progress()):
    progress(0, desc="Starting")
    
    cwd = os.getcwd()
    
    os.chdir(f"{cwd}/codeart/evaluation/binary-similarity")
    progress(0.1, desc="Starting")
    
    command = ["python3", "inference.py",
                        "--model_name_or_path", model_name,
                        "--masking_enable_global_memory_patterns", "true",
                        "--masking_enable_bridge_patterns", "false",
                        "--masking_enable_graph_patterns", "true",
                        "--masking_enable_local_patterns", "true",
                        "--with_transitive_closure", "true",
                        "--position_embedding_type", "mixed",
                        "--max_relative_position_embeddings", "8",
                        "--normalize_embed", "true",
                        "--batch_size", "48",
                        "--source_file", f"cache/binary_clone_detection/{dataset_name}-query.jsonl",
                        "--target_file", f"cache/binary_clone_detection/{dataset_name}-pool.jsonl",
                        "--source_embed_save_file", f"output/{dataset_name}-src_{run_alias}.npy",
                        "--target_embed_save_file", f"output/{dataset_name}-tgt_{run_alias}.npy",
                        "--zero_shot", "false",
                        "--top_k", "1"]
    

    print(command)

    output = subprocess.run(command, capture_output=True, text=True)

    progress(0.8, desc="Starting")

    output = subprocess.run(["python3", "sample_and_report.py",
                             "--source_file", f"output/{dataset_name}-src_{run_alias}",
                             "--target_file", f"output/{dataset_name}-tgt_{run_alias}",
                             "--source_id_file", f"cache/binary_clone_detection/{dataset_name}-query.id",
                             "--target_id_file", f"cache/binary_clone_detection/{dataset_name}-pool.id",
                             "--pool_size", str(pool_size)], capture_output=True, text=True)
      
    
    os.chdir(cwd)
    return output.stdout

def run_table3_eval(model_name, run_alias, dataset_name, pool_size, progress=gr.Progress()):
    progress(0, desc="Starting")
    
    cwd = os.getcwd()
    
    os.chdir(f"{cwd}/codeart/evaluation/binary-similarity")
    progress(0.1, desc="Starting")
    
    command = ["python3", "inference.py",
                        "--model_name_or_path", model_name,
                        "--masking_enable_global_memory_patterns", "true",
                        "--masking_enable_bridge_patterns", "false",
                        "--masking_enable_graph_patterns", "true",
                        "--masking_enable_local_patterns", "true" if "wo_local" not in model_name else "false",
                        "--with_transitive_closure", "true" if "wo_trans" not in model_name else "false",
                        "--position_embedding_type", "mixed" if "wo_rel" not in model_name else "absolute",
                        "--max_relative_position_embeddings", "8",
                        "--normalize_embed", "true",
                        "--batch_size", "48",
                        "--source_file", f"cache/binary_clone_detection/{dataset_name}-query.jsonl",
                        "--target_file", f"cache/binary_clone_detection/{dataset_name}-pool.jsonl",
                        "--source_embed_save_file", f"output/{dataset_name}-src_{run_alias}.npy",
                        "--target_embed_save_file", f"output/{dataset_name}-tgt_{run_alias}.npy",
                        "--zero_shot", "false",
                        "--top_k", "1"]
    
    if "max_trans_closure_4" in model_name:
        command += ["--max_transitions", "4"]
    elif "max_trans_closure_6" in model_name:
        command += ["--max_transitions", "6"]

    print(command)

    output = subprocess.run(command, capture_output=True, text=True)

    progress(0.8, desc="Starting")

    output = subprocess.run(["python3", "sample_and_report.py",
                             "--source_file", f"output/{dataset_name}-src_{run_alias}",
                             "--target_file", f"output/{dataset_name}-tgt_{run_alias}",
                             "--source_id_file", f"cache/binary_clone_detection/{dataset_name}-query.id",
                             "--target_id_file", f"cache/binary_clone_detection/{dataset_name}-pool.id",
                             "--pool_size", str(pool_size)], capture_output=True, text=True)
      
    
    os.chdir(cwd)
    return output.stdout


with gr.Blocks() as demo:
    with gr.Tab(label="Welcome"):
        message = """
        # CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking
        This is a demo for the CodeArt model. Please select the task you want to perform from the tabs above.
        """
        gr.Markdown(message)
    with gr.Tab(label="Figure 8"):
        gr.Markdown("This is the demo for Figure 8. It takes 10 minutes to run on binutilsh dataset with 50 pool size.")
        with gr.Row():
            with gr.Column(scale=1.5):
                model_name = gr.Dropdown(["PurCL/codeart-binsim"], label="Select Model", allow_custom_value=False)
                
                dataset_name = gr.Dropdown(["binutilsh", "libcurlh", "libmagickh", "opensslh", "libsqlh", "puttyh"], label="Select Dataset", allow_custom_value=False)
                pool_size = gr.Dropdown([32, 50, 100, 200, 300, 500], label="Pool Size", allow_custom_value=False)
        
                with gr.Row():
                    run_alias = gr.Textbox(label="Run Alias", placeholder="Enter run alias")

                    run_button = gr.Button("Run")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", placeholder="Output will be displayed here")

        def update_alias(model_name, dataset_name, pool_size):
            model = model_name.split("/")[-1]
            alias = f"fig8-{model}-{dataset_name}-{pool_size}"
            return alias

        model_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        dataset_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        pool_size.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)

        
        run_button.click(fn=run_fig8_tab1_eval,
                         inputs=[model_name, run_alias, dataset_name, pool_size],
                         outputs=output)
    with gr.Tab(label="Table 1"):
        gr.Markdown("This is the demo for Table 1. It takes 10 minutes to run on binutilsh dataset with 50 pool size.")
        with gr.Row():
            with gr.Column(scale=1.5):
                model_name = gr.Dropdown(["PurCL/codeart-26m"], label="Select Model", allow_custom_value=False)
                
                dataset_name = gr.Dropdown(["binutilsh", "libcurlh", "libmagickh", "opensslh", "libsqlh", "puttyh"], label="Select Dataset", allow_custom_value=False)
                pool_size = gr.Dropdown([32, 50, 100, 200, 500], label="Pool Size", allow_custom_value=False)
        
                with gr.Row():
                    run_alias = gr.Textbox(label="Run Alias", placeholder="Enter run alias")

                    run_button = gr.Button("Run")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", placeholder="Output will be displayed here")

        def update_alias(model_name, dataset_name, pool_size):
            model = model_name.split("/")[-1]
            alias = f"tab1-{model}-{dataset_name}-{pool_size}"
            return alias

        model_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        dataset_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        pool_size.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)

        
        run_button.click(fn=run_fig8_tab1_eval,
                         inputs=[model_name, run_alias, dataset_name, pool_size],
                         outputs=output)
    with gr.Tab(label="Table 2"):
        with gr.Row():
            with gr.Column(scale=1):
                model_name = gr.Dropdown(list(TABLE_2_MODELS.keys()), label="Select Model", allow_custom_value=False)
                        
                with gr.Row():

                    run_button = gr.Button("Run")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", placeholder="Output will be displayed here")

        run_button.click(fn=run_table2_eval,
                         inputs=[model_name],
                         outputs=output)

    with gr.Tab(label="Table 3"):
        with gr.Row():
            with gr.Column(scale=1):
                model_name = gr.Dropdown(list(TABLE_3_MODELS.keys()), label="Select Model", allow_custom_value=False)
                
                dataset_name = gr.Dropdown(["coreutilsh"], label="Select Dataset", allow_custom_value=False)
                pool_size = gr.Dropdown([100], label="Pool Size", allow_custom_value=False)
        
                with gr.Row():
                    run_alias = gr.Textbox(label="Run Alias", placeholder="Enter run alias")

                    run_button = gr.Button("Run")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", placeholder="Output will be displayed here")

        def update_alias(model_name, dataset_name, pool_size):
            model = model_name.split("/")[-1]
            alias = f"{model}-{dataset_name}-{pool_size}"
            return alias

        model_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        dataset_name.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)
        pool_size.change(update_alias, inputs=[model_name, dataset_name, pool_size], outputs=run_alias)

        
        run_button.click(fn=run_table3_eval,
                         inputs=[model_name, run_alias, dataset_name, pool_size],
                         outputs=output)
        
    with gr.Tab(label="Figure 9"):
        with gr.Row():
            with gr.Column(scale=1):
                optimization_level = gr.Dropdown(list(FIG_9_DATASET.keys()),
                                                 label="Select Optimization Level",
                                                 allow_custom_value=False)
                
                model_name = gr.Dropdown(list(FIG_9_MODELS.values()),
                                         label="Select Model",
                                         allow_custom_value=False,
                                         interactive=False)
                
                run_button = gr.Button("Run")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Output", placeholder="Output will be displayed here")

        def update_model_name(optimization_level):
            if optimization_level in FIG_9_MODELS:
                return FIG_9_MODELS[optimization_level]


        optimization_level.change(update_model_name, inputs=[optimization_level], outputs=model_name)

        run_button.click(fn=run_fig9_eval, inputs=[model_name, optimization_level], outputs=output)


if __name__ == "__main__":
    demo.launch(server_port=47907, share=True)