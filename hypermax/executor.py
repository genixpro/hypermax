import subprocess
import json

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
            process = subprocess.run(['python3', '-c', script], stdout=subprocess.PIPE)
            result = json.loads(process.stdout)

            # print("Finished: ", parameters, "Loss: ", result['accuracy'])

            return result