#pragma once

#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

class EmbeddingStreamer : public StreamerBase {
public:
    /// @brief get_embedding_data is called every time new token is decoded for generating embedding
    /// @param token new token generated
    /// @param sz_in_mem size of embedding data in bytes
    /// @return start address of embedding data array
    virtual ov::Tensor get_embedding(int64_t token) = 0;

    virtual ~EmbeddingStreamer() = default;
};

}  // namespace genai
}  // namespace ov
