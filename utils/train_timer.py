import time 


class TrainTimer:
    """
    A class to keep track of training time and progress.

    Attributes:
    - epochs (int): The number of epochs to train for.
    - start_time (float): The time when training started.
    - epoch_start_time (float): The time when the current epoch started.
    - epoch (int): The current epoch number.
    - batch_start_time (float): The time when the current batch started.
    - duration (float): The total duration of training so far.
    - average_batch_time (float): The average time per batch.
    - iter_per_epoch (int): The number of iterations per epoch.
    - total_iter (int): The total number of iterations for all epochs.
    - current_batch (int): The current batch number.
    - current_iter (int): The current iteration number.
    - avg_per_iter (float): The average time per iteration.
    """

    def __init__(self, epochs, all_data_len, batch_size):
        """
        Initializes a new instance of the TrainTimer class.

        Parameters:
        - epochs (int): The number of epochs to train for.
        - all_data_len (int): The total number of data points in the dataset.
        - batch_size (int): The number of data points per batch.
        """
        self.epochs = epochs
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.epoch = 0
        self.batch_start_time = time.time()
        self.duration = 0
        self.average_batch_time = 0
        self.iter_per_epoch = all_data_len // batch_size
        self.total_iter = self.iter_per_epoch * self.epochs
        self.current_batch = 0
        self.current_iter = 0
        self.avg_per_iter = 0.0

    def batch_start(self):
        self.batch_start_time = time.time()

    def batch_end(self):
        """
        Calculates the duration of a batch, updates the average batch time, and estimates the total training time
        and the estimated end time of the training.

        Returns:
        - batch_time: float, the duration of the current batch
        - estimated_total_time: float, the estimated total time for the training
        - estimated_end_time: float, the estimated end time of the training
        """
        batch_end_time = time.time()
        self.current_iter += 1
        batch_time = batch_end_time - self.batch_start_time
        self.durenation += batch_time
        self.avage_batch_time = self.durenation / self.current_iter
        estimated_total_time = self.avage_batch_time * self.total_iter
        estimated_end_time = (self.total_iter - self.current_iter) * self.avage_batch_time

        return batch_time, estimated_total_time, estimated_end_time

    def epoch_start(self):
        self.epoch_start_time = time.time()

    def epoch_end(self):
        epoch_end_time = time.time()
        self.epoch += 1
        epoch_time = epoch_end_time - self.epoch_start_time
        estimated_total_time = epoch_time * self.epochs
        estimated_end_time = (self.epochs - self.epoch) * epoch_time
        epoch_time = self.time_to_min(epoch_time)
        estimated_total_time = self.time_to_min(estimated_total_time)
        estimated_end_time = self.time_to_min(estimated_end_time)
        return epoch_time , estimated_total_time, estimated_end_time
    
    def time_to_min(self, time):
        return round(time / 60, 2)

