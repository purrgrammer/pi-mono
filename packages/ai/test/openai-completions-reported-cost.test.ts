import { beforeEach, describe, expect, it, vi } from "vitest";
import { complete } from "../src/stream.js";
import type { Model } from "../src/types.js";

// Cost capture: opt-in via `usage: { include: true }`, capture the
// upstream-reported total + per-component split into `usage.reportedCost`.

const mockState = vi.hoisted(() => ({
	chunks: [] as unknown[],
	lastParams: undefined as unknown,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.lastParams = params;
					const chunks = mockState.chunks;
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) yield chunk;
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

function openRouterAuto(): Model<"openai-completions"> {
	return {
		id: "openrouter/auto",
		name: "OpenRouter Auto",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 8192,
	};
}

describe("openai-completions reportedCost", () => {
	beforeEach(() => {
		mockState.chunks = [];
		mockState.lastParams = undefined;
	});

	it("opts into OpenRouter cost accounting via usage: { include: true }", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-1",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 1,
					completion_tokens: 1,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		const params = mockState.lastParams as { usage?: { include?: boolean } };
		expect(params.usage).toEqual({ include: true });
	});

	it("captures upstream-reported total cost into usage.reportedCost.total", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-2",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
					cost: 0.00345,
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toEqual({ total: 0.00345 });
	});

	it("derives reportedCost.total from cost_details when upstream reports cost: 0 with non-zero components", async () => {
		// Some proxies report `usage.cost: 0` while still surfacing the
		// per-component split in `cost_details`. When the billed `cost` is
		// missing or 0, fall back to the components' sum.
		mockState.chunks = [
			{
				id: "chatcmpl-3",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 3467,
					completion_tokens: 2131,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
					cost: 0,
					cost_details: {
						upstream_inference_prompt_cost: 0.017335,
						upstream_inference_completions_cost: 0.053275,
					},
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toEqual({
			total: 0.07061,
			input: 0.017335,
			output: 0.053275,
		});
	});

	it("captures input/output cost split from chunk.usage.cost_details", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-4",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
					cost: 0.00501,
					cost_details: {
						upstream_inference_cost: 0.00501,
						upstream_inference_prompt_cost: 0.000013,
						upstream_inference_completions_cost: 0.004997,
					},
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toEqual({
			total: 0.00501,
			input: 0.000013,
			output: 0.004997,
		});
	});

	it("prefers billed top-level cost over components when both are reported (proxy margin)", async () => {
		// OpenRouter reports both top-level `cost` (billed, may include margin)
		// and `cost_details` (upstream-only inference cost). When they diverge,
		// `total` reflects the billed amount; `input`/`output` keep the upstream
		// split so consumers can compute upstream-only cost via `input + output`.
		mockState.chunks = [
			{
				id: "chatcmpl-margin",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 100,
					completion_tokens: 50,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
					cost: 0.006,
					cost_details: {
						upstream_inference_prompt_cost: 0.001,
						upstream_inference_completions_cost: 0.004,
					},
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toEqual({
			total: 0.006,
			input: 0.001,
			output: 0.004,
		});
	});

	it("leaves reportedCost undefined when upstream reports cost: 0 with no components", async () => {
		// Distinct from the previous case: there `cost: 0` is paired with non-zero
		// `cost_details`, and we sum the components. Here both resolve to 0,
		// so we omit `reportedCost` rather than emit `{ total: 0 }`.
		mockState.chunks = [
			{
				id: "chatcmpl-zero",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 1,
					completion_tokens: 1,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
					cost: 0,
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toBeUndefined();
	});

	it("leaves reportedCost undefined when upstream omits cost", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-omit",
				model: "anthropic/claude-opus-4.7",
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 1,
					completion_tokens: 1,
					prompt_tokens_details: { cached_tokens: 0 },
					completion_tokens_details: { reasoning_tokens: 0 },
				},
			},
		];

		const message = await complete(
			openRouterAuto(),
			{ messages: [{ role: "user", content: "hi", timestamp: Date.now() }] },
			{ apiKey: "test" },
		);

		expect(message.usage.reportedCost).toBeUndefined();
	});
});
