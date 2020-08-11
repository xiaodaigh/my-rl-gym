using Pkg
Pkg.activate("rl-test")
includet("identity-skip.jl")
includet("code.jl")

using ReinforcementLearningEnvironments, ReinforcementLearningBase, Flux
using StatsBase, Statistics
const γ = 1 # discount rate
const α = 0.0001 # learning rate

function update_params!(ps, state, action, reward)
    grad = gradient(()->reward*log(eps(Float32) + policy(state)[action]), ps)
    for p in ps
        Flux.update!(p, -α*grad[p])
    end
end

env = CartPoleEnv(max_steps=1000)



action_space = get_action_space(env)

policy = Chain(
    Dense(4, 16, relu),
    Dense(16, 16, relu),
    Dense(16, length(action_space)),
    softmax
)

ps = Flux.params(policy)

obs = observe(env)
policy(obs.state)
update_params!(ps, obs.state, 1, 1)
policy(obs.state)
update_params!(ps, obs.state, 1, -1)
policy(obs.state)
update_params!(ps, obs.state, 2, 1)
policy(obs.state)
update_params!(ps, obs.state, 2, -1)

function ok(n)
    all_states = []
    all_rewards = []
    all_actions = []
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    for i in 1:n
        policy(obs.state)
        states, rewards, actions = rollout(policy; epsilon=epsilon, max_steps=1000)

        length(rewards)

        # train the network
        for i in length(rewards)-1:-1:1
            rewards[i] += rewards[i+1]*γ
        end

        all_actions = vcat(all_actions, actions)
        all_rewards = vcat(all_rewards, rewards)
        all_states = vcat(all_states, states)

        epsilon = max(epsilon_min, epsilon*epsilon_decay)
    end

    all_rewards .= (all_rewards .- mean(all_rewards))./std(all_rewards)

    # randomise the order
    s = sample(1:length(all_rewards), length(all_rewards), replace=false)

    all_states .= all_states[s]
    all_rewards .= all_rewards[s]
    all_actions .= all_actions[s]

    for (s, a, r) in zip(all_states, all_actions, all_rewards)
        update_params!(ps, s, a, r)
    end

    # play it  to show how well it does
    # [length(rollout(policy; epsilon=0.0, max_steps=1000)[3]) for j in 1:1000] |> mean |> println
    # [length(rollout(policy; epsilon=0.0, max_steps=1000)[3]) for j in 1:1000] |> extrema |> println

end

function playone()
    env = CartPoleEnv(max_steps=1000)
    action_space = get_action_space(env) |> collect
    while true
        local action
        _, action = findmax(policy(env.state))
        env(action)
        render(env)
        if get_terminal(observe(env))
            break
        end
        sleep(0.001)
    end
end

@time ok(50)

while true
    @time ok(50)
    if rand() < 0.1
        playone()
    end
end

