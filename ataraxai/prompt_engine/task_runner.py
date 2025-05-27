import datetime
import os



class TaskRunner:
    def __init__(self, llama_interface, context_manager, task_registry):
        self.llama = llama_interface 
        self.context_manager = context_manager 
        self.task_registry = task_registry 

    def _validate_inputs(self, task_def: dict, user_inputs: dict) -> tuple[bool, list[str]]:

        errors = []
        defined_param_names = {param["name"] for param in task_def.get("input_parameters", [])}

        for param_def in task_def.get("input_parameters", []):
            param_name = param_def["name"]
            is_required = param_def.get("required", False)
            expected_type_str = param_def.get("type", "string") # Default to string if not specified

            if param_name in user_inputs:
                user_value = user_inputs[param_name]

                type_ok = False
                if expected_type_str == "string" and isinstance(user_value, str):
                    type_ok = True
                elif expected_type_str == "integer" and isinstance(user_value, int):
                    type_ok = True
                elif expected_type_str == "boolean" and isinstance(user_value, bool):
                    type_ok = True
                elif expected_type_str == "file_content": 
                    type_ok = True 
                elif expected_type_str == "list" and isinstance(user_value, list):
                    type_ok = True
                elif expected_type_str == "dict" and isinstance(user_value, dict):
                    type_ok = True
                elif expected_type_str == "float" and isinstance(user_value, float):
                    type_ok = True
                elif expected_type_str == "datetime" and isinstance(user_value, datetime.datetime):
                    type_ok = True

                if not type_ok:
                    errors.append(f"Parameter '{param_name}' expects type '{expected_type_str}' but got '{type(user_value).__name__}'.")
                    continue 

                rules = param_def.get("validation_rules", {})
                if expected_type_str == "string" and rules.get("must_exist") is True: 
                    if not os.path.exists(user_value):
                        errors.append(f"File path for '{param_name}': '{user_value}' does not exist.")
                if expected_type_str == "integer":
                    if "min_value" in rules and user_value < rules["min_value"]:
                        errors.append(f"Parameter '{param_name}' value {user_value} is less than minimum {rules['min_value']}.")
                    if "max_value" in rules and user_value > rules["max_value"]:
                        errors.append(f"Parameter '{param_name}' value {user_value} is greater than maximum {rules['max_value']}.")

            elif is_required:
                errors.append(f"Required parameter '{param_name}' is missing.")

        for input_name in user_inputs:
            if input_name not in defined_param_names:
                pass 

        return not errors, errors

    def execute(self, task_id, user_inputs):
        task_def = self.task_registry.get_task(task_id)
        if not task_def:
            raise ValueError(f"Task '{task_id}' not found.")

        is_valid, errors = self._validate_inputs(task_def, user_inputs)
        if not is_valid:
            return {"success": False, "errors": errors}

        resolved_context = {}
        for ctx_name in task_def.get("required_context", []):
            resolved_context[ctx_name] = self.context_manager.get_context(ctx_name, user_inputs)

        template_data = {**resolved_context, **user_inputs}

        for param in task_def.get("input_parameters", []):
            if param.get("type") == "file_content" and param["name"] in user_inputs:
                file_path = user_inputs[param["name"]]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data[param["name"] + "_content"] = f.read() 
                except Exception as e:
                    return f"Error reading file {file_path}: {e}"

        prompt_template_str = self.load_prompt_template(task_def["prompt_template_file"])

        if "file_path" in user_inputs and "text_input" not in template_data: 
             if (task_def["task_id"] == "summarize_document_from_path" and
                 "text_input" not in template_data and 
                 "file_path_content" in template_data): 
                 template_data["text_input"] = template_data["file_path_content"]


        final_prompt = prompt_template_str.format(**template_data)


        raw_output = self.llama.generate(final_prompt)

   
        processed_output = raw_output 


        return processed_output

    def load_prompt_template(self, template_file_path):
        pass