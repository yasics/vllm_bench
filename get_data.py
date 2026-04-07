import argparse
import glob
import json
import os
import pandas as pd

def robust_read_json(file_path):
    """健壮的JSON文件读取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    # print(f"文件内容类型: {type(content)}")
    
    # 根据内容类型处理
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            # 列表的字典：[{...}, {...}]
            return pd.DataFrame(content)
        else:
            # 普通列表
            return pd.DataFrame(content, columns=['value'])
    elif isinstance(content, dict):
        # 单个字典
        return pd.DataFrame([content])
    else:
        # 其他类型（字符串、数字等）
        return pd.DataFrame([{'value': content}])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="结果文件夹")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.results_dir, "*.json"))
    files += glob.glob(os.path.join(args.results_dir, "*", "*.json"))
    files += glob.glob(os.path.join(args.results_dir, "*", "*", "*.json"))
    
    print_headers = [
        "model_id",
        "gpu_num",
        "tp",
        "pp",
        "test_no",
        "max_concurrency",
        "num_prompts",
        "input_len",
        "output_len",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "duration",
        "mean_ttft_ms",
        "p99_ttft_ms", 
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    ]
    
    results: pd.DataFrame = None
    for file in files:
        raw_df = robust_read_json(file)
        cur_df = raw_df[print_headers].drop_duplicates(keep="last")
        cur_df["gpu_num"] = cur_df["gpu_num"].astype(int)
        cur_df["tgs"] = cur_df["output_throughput"] / cur_df["gpu_num"]
        if results is None:
            results = cur_df
        else:
            results = pd.concat([results, cur_df], axis=0, ignore_index=True)
    
    results = results.sort_values(by=["model_id","gpu_num","input_len","output_len","max_concurrency","test_no"])
    
    # 输出所有case的性能结果
    print(results.to_string(index=False))
    
    # 保存所有case的性能结果
    save_file_path = os.path.join(args.results_dir, "results_summary.csv")
    results.to_csv(save_file_path, index=False)