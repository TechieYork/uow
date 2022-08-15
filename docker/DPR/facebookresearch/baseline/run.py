import argparse
import os
import sys

# arguments parser setting
parser = argparse.ArgumentParser("DPR automation processing")
parser.add_argument("--step",
                    nargs="*",
                    default=["training", "embedding", "evaluation"],
                    choices=["training", "embedding", "evaluation"])
parser.add_argument("--dpr",
                    nargs="?",
                    default=os.path.join(os.getcwd(), "DPR"))

# path definition
dpr_root = ""
dpr_outputs_path = ""
model_dest_path = ""
embedding_dest_path = ""
evaluation_dest_path = ""
result_path = ""

# command format for each step
# training
train_command_format = '''python {dpr_root}/train_dense_encoder.py \
train_datasets=[nq_train] dev_datasets=[nq_dev] train=biencoder_local output_dir={output_dir}
'''

# embedding
embedding_command_format = '''python {dpr_root}/generate_dense_embeddings.py \
model_file={model_file} \
ctx_src="dpr_wiki" \
shard_id=0 \
num_shards=50 \
out_file={out_file}
'''

# evaluation
evaluation_command_format = '''python {dpr_root}/dense_retriever.py \
model_file={model_file} \
qa_dataset=nq_test \
ctx_datatsets=[dpr_wiki] \
encoded_ctx_files={encoded_ctx_files} \
out_file={out_file} \
'''


def main(args):
    global dpr_root, dpr_outputs_path, model_dest_path, embedding_dest_path, evaluation_dest_path, result_path
    parsed_args = parser.parse_args(args)

    # set up directory layout
    dpr_root = parsed_args.__dict__["dpr"]
    dpr_outputs_path = os.path.join(dpr_root, "outputs")
    model_dest_path = os.path.join(dpr_root, "model")
    embedding_dest_path = os.path.join(dpr_root, "embedding")
    evaluation_dest_path = os.path.join(dpr_root, "evaluation")
    result_path = os.path.join(dpr_root, "result")
    print(dpr_root)
    print(dpr_outputs_path)
    print(model_dest_path)
    print(embedding_dest_path)
    print(evaluation_dest_path)
    print(result_path)

    # run the steps
    steps = parsed_args.__dict__["step"]
    best_model_file, embedding_file = "", ""
    target_result, top20, top100 = "", "", ""
    for step in steps:
        if step == "training":
            run_training()
        elif step == "embedding":
            best_model_file, embedding_file = run_embedding()
        elif step == "evaluation":
            target_result, top20, top100 = run_evaluation(best_model_file, embedding_file)

    # save result
    save_result(result_path, best_model_file, target_result, top20, top100)


def run_training():
    # run
    command = train_command_format.format(dpr_root=dpr_root, output_dir=model_dest_path)
    os.system(command)


def run_embedding():
    # retrieve best model file
    best_model_file = retrieve_model(dpr_outputs_path)
    if best_model_file is None:
        print("Failed to find the best mode file from log")
        return "", ""
    # output embedding file
    embedding_file = os.path.join(embedding_dest_path, "wiki_passages")

    # run
    command = embedding_command_format.format(
        dpr_root=dpr_root,
        model_file=best_model_file,
        out_file=embedding_file)
    os.system(command)
    return best_model_file, embedding_file


def run_evaluation(best_model_file, embedding_file):
    command = evaluation_command_format.format(
        dpr_root=dpr_root,
        model_file=best_model_file,
        encoded_ctx_files="[\"" + embedding_file + "\"]",
        out_file=evaluation_dest_path)
    os.system(command)

    result, top20, top100 = retrieve_result(dpr_outputs_path)
    return result, top20, top100


def retrieve_model(path):
    # find the latest train_dense_encoder.log file in path
    target_file_name = find_latest_file(path, "train_dense_encoder.log")
    if target_file_name is None:
        print("Failed to find the latest file")
        return
    print("target file name:", target_file_name)

    # retrieve best model file
    target_str = "Training finished. Best validation checkpoint"
    target_file = None

    f = open(target_file_name, "r")
    lines = f.readlines()
    for line in lines:
        position = line.find(target_str)
        if position == -1:
            continue
        target_file = line[position + len(target_str):].strip()
    f.close()
    return target_file


def retrieve_result(path):
    # find the latest dense_retriever.log file in path
    target_file_name = find_latest_file(path, "dense_retriever.log")
    if target_file_name is None:
        print("Failed to find the latest file")
        return "", "", ""
    print("target file name:", target_file_name)

    # retrieve final evaluation result
    target_str = "Validation results: top k documents hits accuracy"
    target_result = None
    top20 = None
    top100 = None

    f = open(target_file_name, "r")
    lines = f.readlines()
    for line in lines:
        position = line.find(target_str)
        if position == -1:
            continue
        target_result = line[position + len(target_str):].strip().lstrip("[").rstrip("]")
        accuracies = target_result.split(",")
        top20 = accuracies[20 - 1]
        top100 = accuracies[100 - 1]
    f.close()
    return target_result, top20, top100


def find_latest_file(path, name):
    # walk to get the latest file_name
    target_file_name = None
    file_modify_time = 0
    for file_path, file_dir, file_names in os.walk(path):
        for file_name in file_names:
            if file_name == name:
                temp_file_name = os.path.join(file_path, file_name)
                if os.path.getmtime(temp_file_name) > file_modify_time:
                    file_modify_time = os.path.getmtime(temp_file_name)
                    target_file_name = temp_file_name
    return target_file_name


def save_result(path, best_model_file, accuracy, top20, top100):
    lines = [
        bytes("Best Model:{file}\n".format(file=best_model_file), encoding='utf-8'),
        bytes("Final Accuracy:{accuracy}\n".format(accuracy=accuracy), encoding='utf-8'),
        bytes("Accuracy Top 20:{top20}\n".format(top20=top20), encoding='utf-8'),
        bytes("Accuracy Top 100:{top100}\n".format(top100=top100), encoding='utf-8')
    ]

    print(lines)
    f = open(os.path.join(path, "result.log"), "wb+")
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
