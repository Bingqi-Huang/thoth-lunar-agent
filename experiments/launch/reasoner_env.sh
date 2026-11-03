# Source before `openral deploy sim` to give the reasoner its LLM gateway.
# Without these the reasoner node stays UNCONFIGURED and deploy_e2e's lifecycle_autostart
# for /openral_reasoner fails after 300 s -- the "error" seen in every Nav2 smoke log.
#
#   source experiments/launch/reasoner_env.sh
#
# Export OPENRAL_REASONER_API_KEY (and OPENRAL_REASONER_ENDPOINT) for your OpenAI-compatible provider first.

export OPENRAL_REASONER_ENDPOINT="${OPENRAL_REASONER_ENDPOINT:-https://YOUR-OPENAI-COMPATIBLE-ENDPOINT/v1}"
export OPENRAL_REASONER_DIALECT="${OPENRAL_REASONER_DIALECT:-openai}"
export OPENRAL_REASONER_MODEL="${OPENRAL_REASONER_MODEL:-DeepSeek-V4.1-Flash}"
# the gateway's thinking models reject tool_choice=required
export OPENRAL_REASONER_TOOL_CHOICE="${OPENRAL_REASONER_TOOL_CHOICE:-auto}"
# thinking models spend budget on reasoning_content before answering
export OPENRAL_REASONER_MAX_TOKENS="${OPENRAL_REASONER_MAX_TOKENS:-16384}"
# httpx picks up this host's SOCKS proxy and dies without `socksio`
unset all_proxy ALL_PROXY
