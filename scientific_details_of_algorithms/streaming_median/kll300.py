#!/usr/bin/python

# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys
from random import randint

class  KLL300:
    def __init__(self):
        self.maxSize = 300
        self.size = 0
        self.capacities = [2, 2, 4, 6, 10, 18, 28, 44, 70, 112]
        self.H = len(self.capacities)
        self.compactors = [Compactor() for _ in range(self.H)]

    def sizef(self):
        return sum([len(c) for c in self.compactors])

    def update(self, item):
        self.compactors[0].append(item)
        self.size += 1
        if self.size >= self.maxSize:
            for h in range(self.H - 1):
                if len(self.compactors[h]) >= self.capacities[h]:
                    newItems = self.compactors[h].compact()
                    self.compactors[h+1].extend(newItems)
                    break
        self.size = self.sizef()
        assert(self.size < self.maxSize)

    def cdf(self):
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
             itemsAndWeights.extend( (item, 2**h) for item in items )
        itemsAndWeights.sort()
        items = [t[0] for t in itemsAndWeights]
        weights = [t[1] for t in itemsAndWeights]
        for i in range(len(weights)-1):
            weights[i+1]+=weights[i]
        totWeight = weights[-1]
        return items, [w/totWeight for w in weights]

class Compactor(list):
    def compact(self):
        self.sort()
        offset = randint(0,1)
        for item in self[offset::2]:
            yield item
        self.clear()
