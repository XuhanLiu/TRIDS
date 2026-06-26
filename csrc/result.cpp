/*
 Copyright (c) 2024-2026 @  Shenzhen Bay Laboratory &
							Peking University &
							Changping Laboratory &
							XtalPi Technologies Co., Ltd

 This code is a part of TRIDS:
 The unified molecular docking framework integrated with deep learning-based site
    binding, sampling and scoring.

 TRIDS is open-source software for molecular docking with PyTorch-based DL models:
 (https://www.github.cn/xuhanliu/trids)

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Author:		Dr. Xuhan Liu
 Email:			xuhanliu@qq.com
 */

#include <openbabel/generic.h>
#include "result.h"

Result::Result(at::Tensor pos_, float score_) : score(score_) {
    setCoord(pos_);
}

void Result::updateMol(OpenBabel::OBMol* mol) {
    mol->SetCoordinates(coord.data());
    auto data = (OpenBabel::OBPairData*)mol->GetData("score");
    if (data != nullptr) {
        data->SetValue(std::to_string(score));
    } else {
        auto label = new OpenBabel::OBPairData();
        label->SetAttribute("score");
        label->SetValue(std::to_string(score));
        mol->SetData(label);
    }
}

float Result::getScore() {
    return score;
}

void Result::setScore(float score_) {
    score = score_;
}

double* Result::getCoord() {
    return coord.data();
}

void Result::setCoord(at::Tensor pos_) {
    auto pos_len = pos_.numel();
    pos_ = pos_.cpu().to(torch::kDouble);
    auto pos_ptr = pos_.data_ptr<double>();
    coord.resize(pos_len);
    std::copy(pos_ptr, pos_ptr + pos_len, coord.data());
}