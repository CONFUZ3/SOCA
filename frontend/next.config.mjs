/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    const backend = process.env.SOCA_BACKEND_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
};

export default nextConfig;
