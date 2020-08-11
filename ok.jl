using DataFrames
df = DataFrame(
    :person=>["bob","phil","nick"],
    :london=>[1,1,0],
    :spain=>[1,0,0],
    Symbol("london,spain")=>[1,1,1])



function row_spread(row)
    dict = Dict{String, Int}()
    for (colname, val) in zip(keys(row), values(row))
        if colname == :person
            continue
        end
        for country in split(string(colname), ",")
            dict[country] = get(dict, country, 0) + val
        end
    end
    new_row = hcat(DataFrame(person = row.person), DataFrame(dict))
    new_row
end


reduce(vcat, row_spread(row) for row in eachrow(df))

b=a[1]


df1 = stack(df, names(df[:, Not(:person)]))

dict = Dict(n=>split(n) for n in names(df))