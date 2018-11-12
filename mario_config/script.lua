target = "234";
target_world = tonumber(string.sub(target, 1, 1));
target_level = tonumber(string.sub(target, 2, 2));
target_area = tonumber(string.sub(target, 3, 3));

prevLives = 2

function contest_done()
    if data.lives < prevLives then
        return true
    end
    prevLives = data.lives

    return false

end;
