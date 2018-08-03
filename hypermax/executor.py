import subprocess
import json
import atexit

class Executor:
    def __init__(self, config):
        self.config = config


    def run(self, parameters):
        # print("Running: ", parameters)
        if self.config['type'] == 'python_function':
            script = "from " + self.config['module'] + " import " + self.config['name'] + "\n"
            script += "import json\n"
            script += "result = " + self.config['name'] + "(" + json.dumps(parameters) + ")\n"
            script += "print(json.dumps(result))"

            process = subprocess.Popen(['python3', '-c', script], stdout=subprocess.PIPE)
            atexit.register(lambda: process.kill())
            process.wait()
            try:
                result = json.load(process.stdout)
                return result
            except json.JSONDecodeError as e:
                return {"status": "failed", "accuracy": 0}
