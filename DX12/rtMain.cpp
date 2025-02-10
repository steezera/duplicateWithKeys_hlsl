// 1. rtMain if branch

if (rtMain.IsValid())
{
    device->BindUAV(&rtMain, 0, cmd);
}
else
{
    device->BindUAV(&unbind, 0, cmd);
}

// 2. SRV to UAV
barrierStack.push_back(GPUBarrier::Image(&rtMain, rtMain.desc.layout, ResourceState::UNORDERED_ACCESS));

// 3. BarrierStackFlush
BarrierStackFlush(cmd);

// 4. unbind u0 slot
device->BindUAV(&unbind, 0, cmd);

// 5. UAV to SRV
barrierStack.push_back(GPUBarrier::Image(&rtMain, ResourceState::UNORDERED_ACCESS, rtMain.desc.layout));
