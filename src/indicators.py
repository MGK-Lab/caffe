import numpy as np

class PerformanceIndicators:
    def __init__(self, pred, ref,min_depth = 0.05):
        self.pred_org = np.array(pred)
        self.ref_org = np.array(ref)
        self.pred = np.where(self.pred_org >= min_depth, True, False)
        self.ref = np.where(self.ref_org >= min_depth, True, False)
        self.indicators = {
            'hr': self.hit_rate(),
            'far': self.false_alarm_rate(),
            'csi': self.critical_success_index(),
            'rmse': self.root_mean_square_error(),
            'nse': self.nash_sutcliffe_efficiency()
        }

    def hit_rate(self):
        hits = np.sum((self.pred == True) & (self.ref == True))
        return hits / np.sum(self.ref == True)

    def false_alarm_rate(self):
        false_alarms = np.sum((self.pred == True) & (self.ref == False))
        hits = np.sum((self.pred == True) & (self.ref == True))
        return false_alarms / (hits+false_alarms)

    def critical_success_index(self):
        hits = np.sum((self.pred == True) & (self.ref == True))
        false_alarms = np.sum((self.pred == True) & (self.ref == False))
        misses = np.sum((self.pred == False) & (self.ref == True))
        return hits / (hits + false_alarms + misses)

    def root_mean_square_error(self):
        error = (self.pred_org - self.ref_org)**2
        return np.sqrt(np.mean(error))

    def nash_sutcliffe_efficiency(self):
        mean_ref = np.mean(self.ref_org)
        numerator = np.sum((self.pred_org - self.ref_org)**2)
        denominator = np.sum((self.ref_org - mean_ref)**2)
        return 1 - numerator / denominator
    
    def get_indicators(self):
        return self.indicators
    
    def print_indicators(self):
        for key, value in self.indicators.items():
            print(f"{key}: {value}")
