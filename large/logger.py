import torch
import wandb
import csv
import os
from datetime import datetime

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, args=None):
        self.args = args
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 7
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    @staticmethod
    def get_results_string(best_result):
        result_string = ''
        r = best_result[:, 0]
        result_string += f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 1]
        result_string += f'Highest Test: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 2]
        result_string += f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 3]
        result_string += f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 4]
        result_string += f'   Final Test: {r.mean():.2f} ± {r.std():.2f}'

        return result_string

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            # Ensure all elements are tensors and convert them properly
            result = [torch.tensor(r) * 100 if isinstance(r, (float, int)) else torch.tensor(r) * 100 for r in self.results[run]]
            result = torch.stack(result)  # Stack the list of tensors into a single tensor

            if self.args.save_whole_test_result:
                now = datetime.now()
                _month_day = now.strftime("%m%d")
                timestamp = now.strftime("%m%d-%H%M%S")
                results_path = mkdirs(f'results/runs/{self.args.dataset}/{_month_day}/{self.args.wandb_name}')
                with open(f'{results_path}/{run}-{self.args.run_id}-results.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write the header (optional)
                    writer.writerow(["Epoch", "Train Acc", "Val Acc", "Test Acc", "Val Loss"])

                    # Write the data
                    for epoch in range(len(self.results[run])):
                        # Add 1 to run and epoch indices to match with human counting
                        formatted_row = ['{:.4f}'.format(float(x)) for x in self.results[run][epoch]]
                        writer.writerow([epoch * self.args.eval_step] + formatted_row)
                    # Write the args
                    writer.writerow([])
                    writer.writerow(["Args"])
                    for key, value in vars(self.args).items():
                        writer.writerow([key, value])

                    print(f"Saved results to {self.args.dataset}-{self.args.wandb_name}-{run}-results.csv")

            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print('==========================')
            print_str1 = f'>> Run {run + 1:02d}:\n' + \
                        f'\t Highest Train: {result[:, 0].max():.2f} ' + \
                        f'\t Highest Valid: {result[:, 1].max():.2f} ' + \
                        f'\t Highest Test: {result[:, 2].max():.2f}\n' + \
                        f'\t Chosen epoch based on Valid loss: {argmin * self.args.eval_step} ' + \
                        f'\t Final Train: {result[argmin, 0]:.2f} ' + \
                        f'\t Final Valid: {result[argmin, 1]:.2f} ' + \
                        f'\t Final Test: {result[argmin, 2]:.2f}'
            print(print_str1)

            print_str=f'>> Run {run + 1:02d}:' + \
                f'\t Highest Train: {result[:, 0].max():.2f} ' + \
                f'\t Highest Valid: {result[:, 1].max():.2f} ' + \
                f'\t Highest Test: {result[:, 2].max():.2f}\n' + \
                f'\t Chosen epoch based on Valid acc: {ind * self.args.eval_step} ' + \
                f'\t Final Train: {result[ind, 0]:.2f} ' + \
                f'\t Final Valid: {result[ind, 1]:.2f} ' + \
                f'\t Final Test: {result[ind, 2]:.2f}'
            print(print_str)
            self.test = result[ind, 2]
        else:
            best_results = []
            max_val_epoch = 0

            for r in self.results:
                r = [torch.tensor(res) * 100 if isinstance(res, (float, int)) else torch.tensor(res) * 100 for res in r]
                r = torch.stack(r)  # Stack the list of tensors into a single tensor
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                    max_val_epoch = r[:, 1].argmax()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()
            # if self.args.use_wandb:
            #     wandb.log({
            #         'Average Highest Train': r.mean().item(),
            #         'Std Highest Train': r.std().item(),
            #         'Average Highest Test': best_result[:, 1].mean().item(),
            #         'Std Highest Test': best_result[:, 1].std().item(),
            #         'Average Highest Valid': best_result[:, 2].mean().item(),
            #         'Std Highest Valid': best_result[:, 2].std().item(),
            #         'Average Final Train': best_result[:, 3].mean().item(),
            #         'Std Final Train': best_result[:, 3].std().item(),
            #         'Average Final Test': best_result[:, 4].mean().item(),
            #         'Std Final Test': best_result[:, 4].std().item()
            #     })
            return self.get_results_string(best_result)

    def save(self, params, results, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"{results}\n")
            file.write(f"{params}\n")
            file.write('=='*50)
            file.write('\n')
            file.write('\n')

import os
def save_result(args, results):
    if args.save_result:
        if not os.path.exists(f'results/{args.dataset}'):
            os.makedirs(f'results/{args.dataset}')
        filename = f'results/{args.dataset}/{args.method}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
                f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")
