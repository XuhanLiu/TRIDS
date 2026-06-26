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

#include <stdexcept>
#include <spdlog/spdlog.h>
#include <openbabel/oberror.h>
#include "log.h"

void OpenBabel::set_verbose(int verbose) {
    if (verbose < 0 || verbose > 2) {
        throw std::runtime_error("verbose must be 0 (error), 1 (warn), or 2 (info)");
    }
    spdlog::set_level(
        verbose == 0 ? spdlog::level::err :
        verbose == 1 ? spdlog::level::warn : spdlog::level::info
    );

    auto& ob_logger = OpenBabel::obErrorLog;
    if (verbose == 0) {
        ob_logger.SetOutputLevel(OpenBabel::obError);
    } else if (verbose == 1) {
        ob_logger.SetOutputLevel(OpenBabel::obWarning);
    } else {
        ob_logger.SetOutputLevel(OpenBabel::obInfo);
    }
}

int OpenBabel::get_verbose() {
    if (spdlog::get_level() == spdlog::level::err) {
        return 0;
    } else if (spdlog::get_level() == spdlog::level::warn) {
        return 1;
    } else {
        return 2;
    }
}
