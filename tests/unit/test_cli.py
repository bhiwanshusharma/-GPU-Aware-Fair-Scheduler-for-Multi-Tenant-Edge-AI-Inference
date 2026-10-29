from src.cli.main import (
    build_parser,
    cmd_info,
    cmd_validate_config,
    cmd_inspect_tenants,
    cmd_smoke_test,
    cmd_simulate,
)


def test_cli_parser_subcommands():
    parser = build_parser()
    
    args_info = parser.parse_args(["info"])
    assert args_info.subcommand == "info"

    args_val = parser.parse_args(["validate-config", "--config-dir", "configs"])
    assert args_val.subcommand == "validate-config"
    assert args_val.config_dir == "configs"

    args_insp = parser.parse_args(["inspect-tenants", "--config", "configs/tenants.yaml"])
    assert args_insp.subcommand == "inspect-tenants"

    args_smoke = parser.parse_args(["smoke-test"])
    assert args_smoke.subcommand == "smoke-test"

    args_sim = parser.parse_args(["simulate", "--quantum", "10.0"])
    assert args_sim.subcommand == "simulate"
    assert args_sim.quantum == 10.0


def test_cli_cmd_handlers():
    parser = build_parser()

    # Test info
    assert cmd_info(parser.parse_args(["info"])) == 0

    # Test validate-config
    assert cmd_validate_config(parser.parse_args(["validate-config", "--config-dir", "configs"])) == 0

    # Test inspect-tenants
    assert cmd_inspect_tenants(parser.parse_args(["inspect-tenants", "--config", "configs/tenants.yaml"])) == 0

    # Test smoke-test
    assert cmd_smoke_test(parser.parse_args(["smoke-test"])) == 0

    # Test simulate
    assert cmd_simulate(parser.parse_args(["simulate", "--quantum", "15.0", "--overhead", "0.0"])) == 0

    # Test simulate with predictor
    assert cmd_simulate(parser.parse_args(["simulate", "--policy", "pdwfq", "--use-predictor"])) == 0


def test_cli_predictor_commands():
    parser = build_parser()
    args_train = parser.parse_args(["train-predictor", "--samples", "100"])
    assert args_train.subcommand == "train-predictor"
    assert args_train.samples == 100

    args_eval = parser.parse_args(["evaluate-predictor"])
    assert args_eval.subcommand == "evaluate-predictor"

