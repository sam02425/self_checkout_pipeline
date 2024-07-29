import shutil
import os

def deploy_model(model_dir, deployment_dir, github_repo):
    shutil.copytree(model_dir, deployment_dir, dirs_exist_ok=True)

    # Push to GitHub
    os.system(f'git -C {github_repo} add .')
    os.system(f'git -C {github_repo} commit -m "Deploy updated model"')
    os.system(f'git -C {github_repo} push')

if __name__ == "__main__":
    deploy_model('models/trained_model', '/path/to/self-checkout/system', '/path/to/github/repo')
