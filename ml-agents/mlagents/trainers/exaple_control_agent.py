from mlagents.trainers.learn import CommandLineOptions


def create_env(sub_id: int, options: CommandLineOptions):
    trainer_config_path = options.trainer_config_path
    curriculum_folder = options.curriculum_folder
    # Recognize and use docker volume if one is passed as an argument
    if not options.docker_target_name:
        model_path = "./models/{run_id}-{sub_id}".format(
            run_id=options.run_id, sub_id=sub_id
        )
        summaries_dir = "./summaries"
    else:
        trainer_config_path = "/{docker_target_name}/{trainer_config_path}".format(
            docker_target_name=options.docker_target_name,
            trainer_config_path=trainer_config_path,
        )
        if curriculum_folder is not None:
            curriculum_folder = "/{docker_target_name}/{curriculum_folder}".format(
                docker_target_name=options.docker_target_name,
                curriculum_folder=curriculum_folder,
            )
        model_path = "/{docker_target_name}/models/{run_id}-{sub_id}".format(
            docker_target_name=options.docker_target_name,
            run_id=options.run_id,
            sub_id=sub_id,
        )
        summaries_dir = "/{docker_target_name}/summaries".format(
            docker_target_name=options.docker_target_name
        )
    trainer_config = load_config(trainer_config_path)
    port = options.base_port + (sub_id * options.num_envs)
    if options.env_path is None:
        port = 5004  # This is the in Editor Training Port
    env_factory = create_environment_factory(
        options.env_path,
        options.docker_target_name,
        options.no_graphics,
        run_seed,
        port,
        options.env_args,
    )
    env = SubprocessEnvManager(env_factory, options.num_envs)