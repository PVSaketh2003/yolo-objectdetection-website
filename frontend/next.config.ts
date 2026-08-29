import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow all tunneling domain origins for mobile phones, iPhones, tablets & remote PCs
  allowedDevOrigins: [
    "sairamsaketh-yolo-studio.loca.lt",
    "*.loca.lt",
    "*.trycloudflare.com",
    "localhost",
    "127.0.0.1"
  ],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8080/api/:path*",
      },
    ];
  },
};

export default nextConfig;
