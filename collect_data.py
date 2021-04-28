import subprocess
import ast


def collect_data():
    training_models = ["baseline"]
    steps = 11
    
    for training_model in training_models:
        for hdr in range(0, steps):
            hdr /= (steps-1)

            cmd = "python3 " + training_model + ".py --hidden_data_ratio " + str(hdr) + " --do_train"
            print("running: " + cmd)

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()


    print("model\thdr\tEM\tF1")
    for training_model in training_models:
        for hdr in range(0, steps):
            hdr /= (steps-1)

            predictions_name = training_model + "_w_hdr_" + str(hdr) + ".txt"
            cmd = "python3 evaluate.py --output_path ./predictions/" + training_model + "_w_hdr_" + str(hdr) + ".txt"

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            for lin in out.decode('utf-8').split('\n'):
                if len(lin) > 0:
                    results = ast.literal_eval(lin)
                    print(training_model + "\t" + str(hdr) + "\t" + str(results["EM"]) + "\t" + str(results["F1"]))


if __name__ == "__main__":
    collect_data()