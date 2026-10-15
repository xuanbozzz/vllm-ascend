#!/usr/bin/env python3
"""读取 mixed_w2w4_moe_ffn_cases.ini，编译一次并顺序运行全部 case。"""

import configparser
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_DIR / "mixed_w2w4_moe_ffn_cases.ini"


def run_command(title, command, env, log_file):
    command_text = " ".join(shlex.quote(str(value)) for value in command)
    message = "\n===== {} =====\n$ {}\n".format(title, command_text)
    print(message, end="", flush=True)

    with log_file.open("a", encoding="utf-8") as output:
        output.write(message)
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
        return process.wait()


def main():
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str.upper
    config.read(CONFIG_FILE, encoding="utf-8")

    if "global" not in config:
        print("Missing [global] in {}".format(CONFIG_FILE), file=sys.stderr)
        return 1

    global_config = config["global"]
    toolkit_home = global_config["ASCEND_TOOLKIT_HOME"]
    result_root = PROJECT_DIR / global_config.get("RESULT_ROOT", "results")
    device_id = global_config.get("DEVICE_ID", "1")
    warmup = global_config.get("WARMUP_CYCLES", "5")
    test = global_config.get("TEST_CYCLES", "100")
    seed = global_config.get("SEED", "1234")

    case_sections = [
        section for section in config.sections() if section.startswith("case:")
    ]
    if not case_sections:
        print("No [case:name] sections found.", file=sys.stderr)
        return 1

    result_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ASCEND_TOOLKIT_HOME"] = toolkit_home

    build_log = result_root / "build.log"
    build_log.write_text("", encoding="utf-8")
    if run_command(
        "build",
        ["bash", PROJECT_DIR / "run_mixed_w2w4_moe_ffn_esl.sh"],
        env,
        build_log,
    ):
        print("Build failed: {}".format(build_log), file=sys.stderr)
        return 1

    for index, section in enumerate(case_sections, start=1):
        case_name = section.split(":", 1)[1].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_name):
            print("Invalid case name: {}".format(case_name), file=sys.stderr)
            return 1

        case = config[section]
        num_experts = case["NUM_EXPERTS"]
        dim_k = case["DIM_K"]
        dim_n = case["DIM_N"]
        dim_d = case["DIM_D"]
        group_list = case["GROUP_LIST"].replace(" ", "")
        is_expert_w4 = case["IS_EXPERT_W4"].replace(" ", "")
        is_down_w4 = case["IS_DOWN_W4"].replace(" ", "")

        route_lists = {
            "GROUP_LIST": group_list,
            "IS_EXPERT_W4": is_expert_w4,
            "IS_DOWN_W4": is_down_w4,
        }
        if any(len(value.split(",")) != int(num_experts) for value in route_lists.values()):
            print(
                "{}: NUM_EXPERTS does not match GROUP_LIST/route lists".format(case_name),
                file=sys.stderr,
            )
            return 1
        if any(value not in {"0", "1"} for value in is_expert_w4.split(",") + is_down_w4.split(",")):
            print("{}: route entries must be 0 or 1".format(case_name), file=sys.stderr)
            return 1
        for expert_id, (gate_route, down_route) in enumerate(
            zip(is_expert_w4.split(","), is_down_w4.split(","))
        ):
            if gate_route == "1" and down_route == "0":
                print(
                    "{}: invalid 4+2 route at expert {}".format(case_name, expert_id),
                    file=sys.stderr,
                )
                return 1

        case_dir = result_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_log = case_dir / "run.log"
        status_file = case_dir / "status.txt"
        case_log.write_text("", encoding="utf-8")
        status_file.write_text("RUNNING\n", encoding="utf-8")

        print(
            "\n######## CASE {}/{}: {} ########".format(
                index, len(case_sections), case_name
            )
        )

        case_env = env.copy()
        case_env["PROFILE_OUTPUT"] = str(case_dir / "profile")
        commands = [
            (
                "generate",
                [
                    sys.executable,
                    PROJECT_DIR / "scripts" / "gen_data.py",
                    num_experts,
                    dim_k,
                    dim_n,
                    dim_d,
                    "--group-list",
                    group_list,
                    "--is-expert-w4",
                    is_expert_w4,
                    "--is-down-w4",
                    is_down_w4,
                    "--seed",
                    seed,
                ],
            ),
            (
                "run",
                [
                    "bash",
                    PROJECT_DIR / "run_mixed_w2w4_moe_ffn_case.sh",
                    num_experts,
                    dim_k,
                    dim_n,
                    dim_d,
                    device_id,
                    warmup,
                    test,
                ],
            ),
            (
                "verify",
                [sys.executable, PROJECT_DIR / "cmp.py", num_experts, dim_d],
            ),
        ]

        for stage, command in commands:
            if run_command(
                "{}:{}".format(case_name, stage),
                command,
                case_env,
                case_log,
            ):
                status_file.write_text("FAIL: {}\n".format(stage), encoding="utf-8")
                print(
                    "{} failed at {}; see {}".format(case_name, stage, case_log),
                    file=sys.stderr,
                )
                return 1

        status_file.write_text("PASS\n", encoding="utf-8")

    print("\nAll {} cases passed.".format(len(case_sections)))
    print("Results: {}".format(result_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
