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

#include <neural-graphics-primitives/common.h>

#include <atomic>
#include <deque>
#include <functional>
#include <future>
#include <thread>
#include <vector>

namespace ngp {

template <typename T>
void wait_all(T&& futures) {
	for (auto& f : futures) {
		f.get();
	}
}

class ThreadPool {
public:
	ThreadPool();
	ThreadPool(size_t maxNum_threads, bool force = false);
	virtual ~ThreadPool();

	template <class F>
	auto enqueue_task(F&& f, bool high_priority = false) -> std::future<std::result_of_t <F()>> {
		using return_type = std::result_of_t<F()>;

		auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));

		auto res = task->get_future();

		{
			std::lock_guard<std::mutex> lock{m_task_queue_mutex};

			if (high_priority) {
				m_task_queue.emplace_front([task]() { (*task)(); });
			} else {
				m_task_queue.emplace_back([task]() { (*task)(); });
			}
		}

		m_worker_condition.notify_one();
		return res;
	}

	void start_threads(size_t num);
	void shutdown_threads(size_t num);
	void set_n_threads(size_t num);

	void wait_until_queue_completed();
	void flush_queue();

	template <typename Int, typename F>
	void parallel_for_async(Int start, Int end, F body, std::vector<std::future<void>>& futures) {
		Int local_num_threads = (Int)m_num_threads;

		Int range = end - start;
		Int chunk = (range / local_num_threads) + 1;

		for (Int i = 0; i < local_num_threads; ++i) {
			futures.emplace_back(enqueue_task([i, chunk, start, end, body] {
				Int inner_start = start + i * chunk;
				Int inner_end = std::min(end, start + (i + 1) * chunk);
				for (Int j = inner_start; j < inner_end; ++j) {
					body(j);
				}
			}));
		}
	}

	template <typename Int, typename F>
	std::vector<std::future<void>> parallel_for_async(Int start, Int end, F body) {
		std::vector<std::future<void>> futures;
		parallel_for_async(start, end, body, futures);
		return futures;
	}

	template <typename Int, typename F>
	void parallel_for(Int start, Int end, F body) {
		wait_all(parallel_for_async(start, end, body));
	}

private:
	size_t m_num_threads = 0;
	std::vector<std::thread> m_threads;

	std::deque<std::function<void()>> m_task_queue;
	std::mutex m_task_queue_mutex;
	std::condition_variable m_worker_condition;
	std::condition_variable m_task_queue_completed_condition;
};

}
