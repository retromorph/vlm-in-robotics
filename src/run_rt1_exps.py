# Import necessary libraries
import os
import time
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from IPython.display import HTML, display
import mediapy
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.rt1.rt1_model import RT1Inference
from llserver.utils.handler import UniserverHandler


class ExperimentRunner:
    def __init__(
            self,
            env_name,
            model_path=None,
            policy_setup=None,
            model_id=None,
            port=8000,
            use_handler=True,
            experiment_name="default_experiment",
            random_seed=42
    ):
        """Initialize the experiment runner.

        Args:
            env_name: Name of the environment to run
            model_path: Path to the saved model
            policy_setup: Policy setup for the model
            model_id: Model ID for the handler
            port: Port for the handler
            use_handler: Whether to use handler or direct model for predictions
            experiment_name: Name of the experiment for logging purposes
        """
        self.env_name = env_name
        self.model_path = model_path
        self.policy_setup = policy_setup
        self.model_id = model_id
        self.port = port
        self.use_handler = use_handler
        self.experiment_name = experiment_name
        self.random_seed = random_seed

        # Setup logging
        self.log_dir, self.session_dir, self.img_dir = self._setup_logging_directory()

        # Initialize components
        self.env, self.obs, self.instruction = self._setup_environment(random_seed=self.random_seed)

        if not self.use_handler and self.model_path:
            self.model = self._setup_model()
            self.model.reset(self.instruction)

        if self.use_handler and self.model_id:
            self.handler = self._setup_handler()

    def _setup_environment(self, random_seed):
        """Initialize and reset the environment."""
        env = simpler_env.make(self.env_name)
        obs, reset_info = env.reset(seed=random_seed)
        instruction = env.get_language_instruction()
        print("Reset info:", reset_info)
        print("Instruction:", instruction)
        return env, obs, instruction

    def _setup_model(self):
        """Initialize the model."""
        model = RT1Inference(saved_model_path=self.model_path, policy_setup=self.policy_setup)
        return model

    def _setup_handler(self):
        """Initialize the handler."""
        handler = UniserverHandler(port=self.port)
        return handler

    def _setup_logging_directory(self):
        """Create and return paths for logging."""
        log_dir = "/home/mpatratskiy/work/SimplerEnv/logs"
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        session_dir = os.path.join(log_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)

        # Create experiment-specific directory
        experiment_dir = os.path.join(session_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        img_dir = os.path.join(experiment_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        return log_dir, experiment_dir, img_dir

    def _save_image(self, image):
        """Save image and return filename."""
        img_timestamp = int(time.time())
        img_filename = f"{img_timestamp}.png"
        img_path = os.path.join(self.img_dir, img_filename)

        pil_image = Image.fromarray(image)
        pil_image.save(img_path)

        return img_filename, pil_image

    def _model_step(self, image):
        """Process a single step with either model or handler."""
        if self.use_handler:
            return self._handler_step(image)
        else:
            return self._direct_model_step(image)

    def _handler_step(self, image):
        """Process a step using the handler."""
        # Save image to server path
        path_to_save = "/home/mpatratskiy/work/meta_world/llserver/data/simplerenv"
        image_path = path_to_save + "/0.png"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)

        # Save the current image for logging
        img_filename, _ = self._save_image(image)

        # API call
        base_path = "/llserver/data/"
        image_paths = [base_path + "simplerenv/0.png"]
        put_response = self.handler.put_task(model_id=self.model_id, prompt=self.instruction, image_paths=image_paths)
        task_id = put_response["task_id"]["task_id"]

        # Wait for task completion
        status = ""
        wait_time = 1
        while status != "completed":
            time.sleep(wait_time)
            result_response = self.handler.get_task_result(model_id=self.model_id, task_id=task_id)
            status = result_response.get("status")

        result = result_response.get("result")
        action = np.array(result["action"])

        # Log data to JSON
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "instruction": self.instruction,
            "action": result["action"],
            "text": result.get("text", ""),
            "image_filename": img_filename,
            "environment": self.env_name
        }

        self._log_to_file(log_entry)

        return action

    def _direct_model_step(self, image):
        """Process a step using the model directly."""
        # Save the current image for logging
        img_filename, _ = self._save_image(image)

        # Get action from model
        raw_action, action = self.model.step(image)

        # Log data to JSON
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "instruction": self.instruction,
            "action": {
                "world_vector": action["world_vector"].tolist(),
                "rot_axangle": action["rot_axangle"].tolist(),
                "gripper": action["gripper"].tolist()
            },
            "image_filename": img_filename,
            "environment": self.env_name
        }

        self._log_to_file(log_entry)
        action_array = np.concatenate([
            action["world_vector"],
            action["rot_axangle"],
            action["gripper"]
        ])
        return action_array

    def _log_to_file(self, log_entry):
        """Log data to a JSON file."""
        log_file = os.path.join(self.session_dir, "log.json")

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []

        log_data.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _update_log_with_info(self, info):
        """Update the last log entry with environment info."""
        log_file = os.path.join(self.session_dir, "log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)

            if log_data:  # Check if log_data is not empty
                # Add info to the last log entry
                log_data[-1]["info"] = info

                # Convert numpy types to Python native types for JSON serialization
                if isinstance(log_data[-1]["info"], dict):
                    for key, value in log_data[-1]["info"].items():
                        if hasattr(value, "item") and isinstance(value, (np.bool_, np.integer, np.floating)):
                            log_data[-1]["info"][key] = value.item()
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                if hasattr(v, "item") and isinstance(v, (np.bool_, np.integer, np.floating)):
                                    value[k] = v.item()

                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)

    def display_images(self, images, save_path=None, do_display=True):
        """Create a GIF from a list of RGB images. Save to the given path if provided, else return HTML to view."""
        fig = plt.figure()
        ims = []

        for image in images:
            im = plt.imshow(image, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

        if save_path:
            ani.save(save_path, writer='imagemagick')
            plt.close(fig)  # Prevents the final frame from being displayed as a static image

        if do_display:
            plt.close(fig)  # Prevents the final frame from being displayed as a static image
            return HTML(ani.to_jshtml())

    def run_experiment(self, max_timesteps):
        """Run the full experiment."""
        print(f"Running experiment with instruction: {self.instruction}")

        # Get initial image
        image = get_image_from_maniskill2_obs_dict(self.env, self.obs)
        images = [image]

        # Run inference loop
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        actions = []

        print(f"Starting experiment loop (max {max_timesteps} steps)")
        while not (predicted_terminated or success or timestep > max_timesteps):
            print(f"Timestep {timestep}")

            # Step the model
            action = self._model_step(image)
            actions.append(action)

            self.obs, reward, success, _, info = self.env.step(action)

            # Update log with environment info
            self._update_log_with_info(info)

            print(f"  Step {timestep}, Success: {success}, Info: {info}")

            # Update image observation
            image = get_image_from_maniskill2_obs_dict(self.env, self.obs)
            images.append(image)

            # Save current frame
            self._save_image(image)

            timestep += 1
        # Print finish reason
        if success:
            finish_reason = "Success"
        elif predicted_terminated:
            finish_reason = "Predicted termination"
        elif truncated:
            finish_reason = "Truncated"
        else:
            finish_reason = "Max timesteps reached"

        print(f"Experiment finished due to: {finish_reason}")
        print(f"Experiment completed after {timestep} steps. Success: {success}")

        # Create and save animation
        gif_path = os.path.join(self.session_dir, f"{self.experiment_name}_{self.env_name}.gif")
        self.display_images(images, save_path=gif_path)
        print(f"Animation saved to {gif_path}")

        return images, actions, success


# Main execution
if __name__ == "__main__":
    # Configuration
    all_envs = [
        # 'google_robot_pick_coke_can',
        # 'google_robot_pick_horizontal_coke_can',
        # 'google_robot_pick_vertical_coke_can',
        # 'google_robot_pick_standing_coke_can',
        # 'google_robot_pick_object',
        # 'google_robot_move_near_v0',
        # 'google_robot_move_near_v1',
        # 'google_robot_move_near',
        # 'google_robot_open_drawer',
        # 'google_robot_open_top_drawer',
        # 'google_robot_open_middle_drawer',
        # 'google_robot_open_bottom_drawer',
        # 'google_robot_close_drawer',
        # 'google_robot_close_top_drawer',
        # 'google_robot_close_middle_drawer',
        # 'google_robot_close_bottom_drawer',
        # 'google_robot_place_in_closed_drawer',
        # 'google_robot_place_in_closed_top_drawer',
        # 'google_robot_place_in_closed_middle_drawer',
        # 'google_robot_place_in_closed_bottom_drawer',
        # 'google_robot_place_apple_in_closed_top_drawer',
        'widowx_spoon_on_towel',
        # 'widowx_carrot_on_plate',
        # 'widowx_stack_cube',
        # 'widowx_put_eggplant_in_basket'
    ]

    experiment_name = "rt1-x-more-spoons"
    # model_id = "5733d5e5-df76-42b5-8bf3-eed829769beb"
    saved_model_path = "/home/mpatratskiy/work/SimplerEnv/checkpoints/rt_1_x_tf_trained_for_002272480_step"
    policy_setup = "widowx_bridge"
    use_handler = False
    random_seeds = [42, 69, 70, 80, 90, 228, 335]
    # Load existing stats if available

    for random_seed in random_seeds:
        # Run experiments for all environments
        for env_name in all_envs:
            print(f"Running experiment {experiment_name} [seed: {random_seed}] for environment: {env_name}")
            full_experiment_name = experiment_name + '_' + env_name + '_' + str(random_seed)
            stats_path = f"/home/mpatratskiy/work/SimplerEnv/logs/stats_{full_experiment_name}.json"

            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {}
            # Create experiment runner
            runner = ExperimentRunner(
                env_name=env_name,
                # model_id=model_id,
                experiment_name=full_experiment_name,  # Use env_name as experiment_name
                model_path=saved_model_path,
                policy_setup=policy_setup,
                use_handler=use_handler,
                random_seed=random_seed
            )

            # Run the experiment
            images, actions, success = runner.run_experiment(max_timesteps=150)

            # Save success status to stats
            stats[experiment_name + '_' + env_name + '_' + str(random_seed)] = success

            # Update stats file after each experiment
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"Updated stats in {stats_path}")
