target = "234";
target_world = tonumber(string.sub(target, 1, 1));
target_level = tonumber(string.sub(target, 2, 2));
target_area = tonumber(string.sub(target, 3, 3));

prevLives = 2


level_max_x = {
-- 1
    ["levelHi=0,levelLo=0"] = 3266,
    ["levelHi=0,levelLo=1"] = 3266,
    ["levelHi=0,levelLo=2"] = 2514,
    ["levelHi=0,levelLo=3"] = 2430,

-- 2
    ["levelHi=1,levelLo=0"] = 3298,
    ["levelHi=1,levelLo=1"] = 3266,
    ["levelHi=1,levelLo=2"] = 3682,
    ["levelHi=1,levelLo=3"] = 2430,

-- 3
    ["levelHi=2,levelLo=0"] = 3298,
    ["levelHi=2,levelLo=1"] = 3442,
    ["levelHi=2,levelLo=2"] = 2498,
    ["levelHi=2,levelLo=2"] = 2430,

-- 4
    ["levelHi=3,levelLo=0"] = 3698,
    ["levelHi=3,levelLo=1"] = 3266,
    ["levelHi=3,levelLo=2"] = 2434,
    ["levelHi=3,levelLo=3"] = 2942,

-- 5
    ["levelHi=4,levelLo=0"] = 3282,
    ["levelHi=4,levelLo=1"] = 3298,
    ["levelHi=4,levelLo=2"] = 2514,
    ["levelHi=4,levelLo=3"] = 2429,

-- 6
    ["levelHi=5,levelLo=0"] = 3106,
    ["levelHi=5,levelLo=1"] = 3554,
    ["levelHi=5,levelLo=2"] = 2754,
    ["levelHi=5,levelLo=3"] = 2429,
-- 7
    ["levelHi=6,levelLo=0"] = 2962,
    ["levelHi=6,levelLo=1"] = 3266,
    ["levelHi=6,levelLo=2"] = 3682,
    ["levelHi=6,levelLo=3"] = 3453,
-- 8
    ["levelHi=7,levelLo=0"] = 6114,
    ["levelHi=7,levelLo=1"] = 3554,
    ["levelHi=7,levelLo=2"] = 3554,
    ["levelHi=7,levelLo=3"] = 4989
    -- ["zone=5,act=2"] = 000000, -- does not have a max x
}


function contest_done()

    -- agent is dead
    if (data.player_state == 6) or (data.player_state == 11) then
        return true
    end

    -- agent is losing its lives
    if data.lives < prevLives then
        return true
    end
    prevLives = data.lives

    -- reach the goal of the level
    if calc_progress(data) >= 1 then
       return true
    end

    return false

end;

function level_key()
    return string.format("levelHi=%d,levelLo=%d", data.levelHi, data.levelLo)
end

data.offset_x = nil
end_x = nil

function clip(v, min, max)
    if v < min then
        return min
    elseif v > max then
        return max
    else
        return v
    end
end

function calc_progress(data)
    if data.offset_x == nil then
        data.offset_x = -data.x
        end_x = level_max_x[level_key()] - data.x
    end

    local cur_x = clip(data.curr_page * 256 + data.x + data.offset_x, 0, end_x)
    return cur_x / end_x
end

data.prev_progress = 0
function contest_reward()
    local progress = calc_progress(data)
    local reward = (progress - data.prev_progress) * 9000
    data.prev_progress = progress

    return reward
end
