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

#pragma once

#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

struct parse_error {
	fs::path file;
	unsigned line;
	string reason;
	parse_error() : line(0) {}
	parse_error(unsigned line_, const string& reason_ = "") : line(line_), reason(reason_) {}
	parse_error(const fs::path& file_, unsigned line_, const string& reason_ = "")
		: file(file_), line(line_), reason(reason_) {}
};

struct stream_parse_error : public parse_error {

	stream_parse_error(unsigned line_, const string& reason_)
		: parse_error(line_, reason_) {}
	parse_error to_parse_error(const fs::path& name) const {
		return parse_error(name, line, reason);
	}
};

struct internal_error {
	string file;
	unsigned line;
	internal_error(const string& file_, unsigned line_) : file(file_), line(line_) {}
};


struct numerical_error : public runtime_error {
	numerical_error(const string& message) : runtime_error(message) {}
};

struct usage_error : public runtime_error {
	usage_error(const string& message) : runtime_error(message) {}
};


//thrown when can't parse name of term
struct scoring_error {
	string name;
	string msg;
	scoring_error(const string& n, const string& m = "") : name(n), msg(m) {}
};

struct file_error {
	fs::path name;
	bool in;
	file_error(const fs::path& name_, bool in_) : name(name_), in(in_) {}
};

struct syntax_error {
	std::string nature;
	syntax_error(const std::string& nature_)
		: nature(nature_) {}
};
