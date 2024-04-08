import os
import yaml
import jinja2
import argparse

# build script using jinja2 template
def build_script(template_path: str, 
                 template_file: str, 
                 output_file: str, 
                 **kwargs):
    template_loader = jinja2.FileSystemLoader(searchpath=template_path)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_file)
    output = template.render(**kwargs)
    with open(output_file, "w") as f:
        f.write(output)

# python ,main entry point 
if __name__ == "__main__":

    # parse main args
    parser = argparse.ArgumentParser()
    # add argument for template file
    parser.add_argument("--template_file", type=str, required=True)

    # add argument for config file
    parser.add_argument("--config_file", type=str, required=True)

    # add argument for output file
    parser.add_argument("--output_file", type=str, required=True)

    # user parser to parse args
    args = parser.parse_args()

    from pathlib import Path

    file = Path(__file__).resolve()
    base_dir = file.parent

    # safe load yaml config file
    with open(os.path.join(base_dir, args.config_file), "r") as f:
        config = yaml.safe_load(f)

    build_script(template_path=os.path.join(base_dir, "templates"), 
                 template_file=args.template_file, 
                 output_file=os.path.join(base_dir, args.output_file), 
                 **config)


