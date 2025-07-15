import numpy as np
import torch
from verl import DataProto
from verl.protocol import list_of_dict_to_dict_of_list
import copy
from tensordict import TensorDict


class TrailDataProto(DataProto):
    def select_by_index(self, idx):
        """Select a single index from the DataProto. Alias for __getitem__."""
        return self[idx]

    @classmethod
    def from_list(cls, data_list: list):
        """Create a DataProto from a list of DataProtoItems."""
        if not data_list:
            return cls()

        batch_list = [item.batch for item in data_list if item.batch is not None]
        non_tensor_list = [item.non_tensor_batch for item in data_list]

        new_batch_td = None
        if batch_list:
            # Convert list of dictionaries to a dictionary of lists of tensors
            batch_of_lists = list_of_dict_to_dict_of_list(batch_list)
            # Stack tensors in each list to form a batch tensor
            new_batch = {k: torch.stack(v) for k, v in batch_of_lists.items()}
            batch_size = len(data_list)
            new_batch_td = TensorDict(source=new_batch, batch_size=[batch_size])

        non_tensor_batch = {}
        if non_tensor_list:
            # Deepcopy to prevent memory leaks from holding references to old objects
            non_tensor_batch = copy.deepcopy(list_of_dict_to_dict_of_list(non_tensor_list))
            for key, val in non_tensor_batch.items():
                if val and not isinstance(val[0], np.ndarray):
                    non_tensor_batch[key] = np.array(val, dtype=object)
                elif val:
                    try:
                        non_tensor_batch[key] = np.array(val)
                    except ValueError:
                        # Handle cases with inconsistent shapes by using dtype=object
                        non_tensor_batch[key] = np.array(val, dtype=object)

        meta_info = data_list[0].meta_info if data_list else {}
        return cls(batch=new_batch_td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
