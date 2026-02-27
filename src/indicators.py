# ------------------------------------------------------------------------------
# Dynamic CA-ffe
# Copyright (C) 2022–2026 Maziar Gholami Korzani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

import numpy as np


class PerformanceIndicators:
    def __init__(self, pred, ref, min_depth=0.05):
        self.threshold = min_depth
        self.pred_org = np.array(pred)
        self.ref_org = np.array(ref)
        self.pred = np.where(self.pred_org > self.threshold, True, False)
        self.ref = np.where(self.ref_org > self.threshold, True, False)
        self.hit_rate = 0
        self.false_alarm_rate = 0
        self.critical_success_index = 0
        self.root_mean_square_error = 0
        self.nash_sutcliffe_efficiency = 0
        self.r_squared = 0

        # self.indicators = {
        #     'hr': self.hit_rate(),
        #     'far': self.false_alarm_rate(),
        #     'csi': self.critical_success_index(),
        #     'rmse': self.root_mean_square_error(),
        #     'nse': self.nash_sutcliffe_efficiency()
        # }

    def CalculateIndicators(self):
        TP = np.sum((self.pred == True) & (self.ref == True)) * 1.
        FN = np.sum((self.pred == False) & (self.ref == True)) * 1.
        FP = np.sum((self.pred == True) & (self.ref == False)) * 1.

        self.hit_rate = TP / (TP + FN)
        self.false_alarm_rate = FP / (TP + FP)
        self.critical_success_index = TP / (TP + FN + FP)

        ma = (self.pred == True) | (self.ref == True)
        n = np.sum(ma) * 1.
        ss = np.sum(np.square(self.ref_org[ma] - self.pred_org[ma]))

        self.root_mean_square_error = np.sqrt(ss / n)

        mean_ref = np.mean(self.ref_org[ma])
        ss2 = np.sum(np.square(self.ref_org[ma] - mean_ref))

        self.nash_sutcliffe_efficiency = 1 - (ss / ss2)

        ma_2 = (self.ref == True)
        Y_bar_b = np.mean(self.ref_org[ma_2])
        self.r_squared = 1 - ss / np.sum(
            np.square(self.ref_org[ma_2] - Y_bar_b))

        self.indicators = {
            'threshold': self.threshold,
            'hr': self.hit_rate,
            'far': self.false_alarm_rate,
            'csi': self.critical_success_index,
            'rmse': self.root_mean_square_error,
            'nse': self.nash_sutcliffe_efficiency,
            'rs': self.r_squared
        }

    def print_indicators(self):
        for key, value in self.indicators.items():
            print(f"{key}: {value}")
