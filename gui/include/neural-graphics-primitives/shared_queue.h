/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file was taken from the tev image viewer and is re-released here
// under the NVIDIA Source Code License with permission from the author.

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

namespace ngp {

class ICallable {
public:
	virtual ~ICallable() {}
	virtual void operator()() = 0;
};

template <typename T>
class Callable : public ICallable {
public:
	Callable() = default;
	Callable(const T& callable) : m_callable{callable} {}
	Callable(T&& callable) : m_callable{std::forward<T>(callable)} {}
	Callable(const Callable& other) = delete;
	Callable& operator=(Callable&& other) { std::swap(m_callable, other.m_callable); return *this; }
	Callable(Callable&& other) { *this = std::move(other); }

	void operator()() override {
		m_callable();
	}
private:
	T m_callable;
};

template <typename T>
std::unique_ptr<ICallable> callable(T&& callable) {
	return std::make_unique<Callable<T>>(std::forward<T>(callable));
}

class SharedQueueEmptyException {};

template <typename T>
class SharedQueue {
public:
	bool empty() const {
		std::lock_guard<std::mutex> lock{mMutex};
		return mRawQueue.empty();
	}

	size_t size() const {
		std::lock_guard<std::mutex> lock{mMutex};
		return mRawQueue.size();
	}

	void push(T&& newElem) {
		std::lock_guard<std::mutex> lock{mMutex};
		mRawQueue.emplace_back(std::forward<T>(newElem));
		mDataCondition.notify_one();
	}

	void clear() {
		std::lock_guard<std::mutex> lock{mMutex};
		mRawQueue.clear();
	}

	void clearAndPush(T&& newElem) {
		std::lock_guard<std::mutex> lock{mMutex};
		mRawQueue.clear();
		mRawQueue.emplace_back(std::forward<T>(newElem));
		mDataCondition.notify_one();
	}

	T waitAndPop() {
		std::unique_lock<std::mutex> lock{mMutex};

		while (mRawQueue.empty()) {
			mDataCondition.wait(lock);
		}

		T result = std::move(mRawQueue.front());
		mRawQueue.pop_front();

		return result;
	}

	T tryPop(bool back = false) {
		std::unique_lock<std::mutex> lock{mMutex};

		if (mRawQueue.empty()) {
			throw SharedQueueEmptyException{};
		}

		if (back) {
			T result = std::move(mRawQueue.back());
			mRawQueue.pop_back();
			return result;
		} else {
			T result = std::move(mRawQueue.front());
			mRawQueue.pop_front();
			return result;
		}
	}

private:
	std::deque<T> mRawQueue;
	mutable std::mutex mMutex;
	std::condition_variable mDataCondition;
};

}
