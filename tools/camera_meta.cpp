// Read timestamped metadata (including temperature sensors) from an Okulo camera.
#include <Vidu_SDK_Api.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

int main()
{
  PDdevice device;
  if (!device.init())
  {
    std::cerr << "device init failed" << std::endl;
    return 1;
  }
  std::cout << "device SN: " << device.getDeviceSN() << std::endl;

  for (const char *stream_name : { "RGB", "ToF" })
  {
    std::cout << "\n=== " << stream_name << " ===" << std::endl;
    PDstream stream(device, stream_name);
    if (!stream.init())
    {
      std::cout << "stream init failed" << std::endl;
      continue;
    }

    std::shared_ptr<PDbuffer> frame;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline)
    {
      frame = stream.waitFrames(1);
      if (frame)
      {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!frame)
    {
      std::cout << "no frame" << std::endl;
      continue;
    }
    std::cout << "frame images: " << frame->getImageNum() << std::endl;

    const auto features = stream.GetFeatureList();
    std::cout << "features: " << features.size() << std::endl;
    for (const auto &feature : features)
    {
      if (feature.name_.find("Temp") != std::string::npos)
      {
        std::cout << "  temp feature: " << feature.name_ << " type=" << feature.type_ << std::endl;
      }
    }
    for (const char *key : { "TempSensor", "TempVcsel", "DistCorrEqvTemp" })
    {
      float value = 0.0f;
      const bool ok = stream.get(key, value);
      std::cout << "  feature get " << key << ": ok=" << ok << " value=" << value << std::endl;
    }

    size_t count = 0;
    if (GenTL::PDBufferGetMetaNum(frame->getPort(), &count) == GenTL::GC_ERR_SUCCESS)
    {
      std::cout << "metadata items: " << count << std::endl;
      for (size_t i = 0; i < count; ++i)
      {
        const char *name = nullptr;
        const char *desc = nullptr;
        if (GenTL::PDBufferGetMetaDesc(frame->getPort(), i, &name, &desc) == GenTL::GC_ERR_SUCCESS)
        {
          std::cout << "  [" << i << "] " << (name ? name : "?") << " : " << (desc ? desc : "") << std::endl;
        }
      }
    }
    else
    {
      std::cout << "metadata count unavailable" << std::endl;
    }

    unsigned char raw[256];
    for (size_t i = 0; i < count; ++i)
    {
      const char *name = nullptr;
      const char *desc = nullptr;
      GenTL::PDBufferGetMetaDesc(frame->getPort(), i, &name, &desc);
      size_t size = sizeof(raw);
      GenTL::INFO_DATATYPE type = GenTL::INFO_DATATYPE_UNKNOWN;
      const auto rc = GenTL::PDBufferGetMeta(frame->getPort(), i, &type, raw, &size);
      std::cout << "  get[" << i << "] " << (name ? name : "?") << " rc=" << rc
                << " type=" << type << " size=" << size;
      if (rc == GenTL::GC_ERR_SUCCESS && size == 4)
      {
        float value = 0.0f;
        std::memcpy(&value, raw, sizeof(value));
        std::cout << " float=" << value;
      }
      std::cout << std::endl;
    }

    for (const char *key : { "Range", "TempSensor", "TempVcsel", "DistCorrEqvTemp" })
    {
      float value = 0.0f;
      size_t size = sizeof(value);
      GenTL::INFO_DATATYPE type = GenTL::INFO_DATATYPE_UNKNOWN;
      const auto rc = GenTL::PDBufferGetMetaByName(frame->getPort(), key, &type, &value, &size);
      std::cout << key << ": rc=" << rc << " type=" << type << " size=" << size << " value=" << value
                << std::endl;
    }
  }
  return 0;
}
