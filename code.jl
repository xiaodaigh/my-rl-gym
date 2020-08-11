export rollout

using StatsBase: wsample

function rollout(policy; epsilon = 0.1, kwargs...)
    rewards = Float32[]
    state = Vector{Float32}[]
    actions = Int[]

    env = CartPoleEnv(;kwargs...)


    action_space = get_action_space(env)

    action_spacec = collect(action_space)

    obs = observe(env)

    while true
        push!(state, obs.state)
        if rand() < epsilon
            action = rand(action_space)
        else
            pos = policy(obs.state)
            if any(isnan, pos)
                error("wtf")
            end
            action = wsample(action_spacec, pos)
        end
        push!(actions, action)
        env(action)
        obs = observe(env)
        push!(rewards, obs.reward)

        get_terminal(obs) && break
    end
    state, rewards, actions
end