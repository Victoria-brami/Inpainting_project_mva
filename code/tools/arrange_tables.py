import yaml
import os


def merge_yaml_files(path, layer_name):
    """ Merges all the fid evaluations files into a single one """
    merged_yaml_file = {'feats': {'fid': {}}}
    for file in os.listdir(path):
        if file.startswith(layer_name):
            with open(os.path.join(path, file), 'r') as yfile:
                string = yfile.read()
                data = yaml.load(string, yaml.loader.BaseLoader)
                canal_name = '{}_{}'.format(file.split('_')[2], file.split('_')[3])
                merged_yaml_file['feats']['fid'][canal_name] = [data['feats']['fid']]
    with open(os.path.join(path, '{}_evaluation_metrics_all.yaml'.format(layer_name)), "w") as res_yfile:
        yaml.dump(merged_yaml_file, res_yfile)

if __name__ == '__main__':
    merge_yaml_files('../../evaluation', 'layer_1')
    merge_yaml_files('../../evaluation', 'layer_2')
    merge_yaml_files('../../evaluation', 'layer_3')
    merge_yaml_files('../../evaluation', 'layer_4')
    merge_yaml_files('../../evaluation', 'layer_10')
    merge_yaml_files('../../evaluation', 'layer_14')
    merge_yaml_files('../../evaluation', 'layer_15')
    merge_yaml_files('../../evaluation', 'layer_16')