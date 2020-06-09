import os
from dataclasses import dataclass
import shutil
import stat
import yaml
import time
import subprocess
import psutil

PYTHON_WORKING_DIRECTORY = os.getcwd()

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def load_yaml(path):
    data = None
    with open(path) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return data

def validate_clone(f):
    def wrapper(*args, **kwargs):
        if args[0].is_clone:
            return f(*args, **kwargs)
        else:
            raise Exception("Repository has not yet been cloned.")
    return wrapper

def reset_working_state(f):
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        os.chdir(PYTHON_WORKING_DIRECTORY)
        return res
    return wrapper

@dataclass
class BenchMark:
    repository: str
    name: str
    folder: str = '.tmp/'
    tag: str = None
    
    is_clone: bool = False

    exec_time = 0


    @staticmethod
    def new(config_path):
        # Validate Config File
        config = load_yaml(config_path)

        required = ["create", "setup", "run"]
        if config is None or not all([k in config for k in required]):
            raise Exception("Missing Configuration")

        clone_step = config["create"]
        tag = None
        folder = '.tmp/'

        if 'tag' in clone_step:
            tag = clone_step['tag']
        if 'folder' in clone_step:
            folder = clone_step['folder']

        
        bench = None

        if 'local' in clone_step:
            bench = BenchMark._copy(clone_step['name'], clone_step['local'], folder)
        else:
            bench = BenchMark._clone(clone_step['name'], clone_step['repository'], tag, folder)

        bench._add_step(config['setup'], 'setup')
        bench._add_step(config['run'], 'run')

        return bench

    @staticmethod
    def _copy(name, local_directory, folder='.tmp/'):
        print(f'=== BENCH {name} | COPY {local_directory} ===')
        try:
            os.mkdir(folder)
        except:
            pass
        
        path = os.path.join(folder, name)
        copytree(local_directory, path)

        return BenchMark(None, name, os.path.join(folder, name), None, True)

    
    @staticmethod
    def _clone(name, repository, tag=None, folder='.tmp/'):
        print(f'=== BENCH {name} | CLONING {repository} ===')
        try:
            os.mkdir(folder)
        except:
            pass

        path = os.path.join(folder, name)

        subprocess.run(f'git clone {repository} {path}', shell=True, stdout=subprocess.DEVNULL)

        if tag:
            os.chdir(path)
            subprocess.run(f'git checkout {tag}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chdir(PYTHON_WORKING_DIRECTORY)
        
        return BenchMark(repository, name, os.path.join(folder, name), tag, True)


    def _add_step(self, steps, _type):
        if _type == 'setup':
            self.setup_steps = steps
        elif _type == 'run':
            self.run_steps = steps
    
    
    def _execute_main_program(self, cmd):
        psutil.cpu_percent(interval=None) # Discarding value
                
        start = time.time()
        subprocess.run(cmd, shell=True)
        end = time.time()

        self.cpu_usage = psutil.cpu_percent(interval=None)
        self.exec_time = end - start
        self.memory_usage = psutil.virtual_memory()

    @reset_working_state
    def _exec_step(self, steps, exec_name=None, args_name=None):
        for step in steps:
            key, val = list(step.items())[0]
            if key == 'enter':
                if val == 'FOLDER':
                    os.chdir(self.folder)
                else:
                    os.chdir(val)
            if key == 'cmd':
                subprocess.run(val, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if key == 'runcmd' and exec_name:
                cmd = ''
                
                if val != '':
                    cmd = f'{val} {exec_name} {args_name}'
                else:
                    cmd = f'.\\{exec_name} {args_name}'
                
                self._execute_main_program(cmd)


            

    @validate_clone
    def setup(self):
        print(f'=== BENCH {self.name} | SETUP ===')
        try:
            steps = self.setup_steps['steps']
            self._exec_step(steps)
        except:
            return False
        return True

    @validate_clone
    def run(self):
        print(f'=== BENCH {self.name} | RUN ===')
        try:
            exec_name = self.run_steps['exec']
            args = self.run_steps['args']
            steps = self.run_steps['steps']

            self._exec_step(steps, exec_name, args)

        except:
            return False
        return True

    @validate_clone
    def clean(self):
        os.chmod(self.folder, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for root, dirs, files in os.walk(self.folder):  
            for ele in dirs:  
                os.chmod(os.path.join(root, ele), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            for ele in files:
                os.chmod(os.path.join(root, ele), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        shutil.rmtree(self.folder)

if __name__ == '__main__':
    """
    test = BenchMark.new('configs/cpu.yaml')

    test.setup()
    test.run()

    test.clean()

    print(f'{test.name}: {test.exec_time}')
    """
    pass