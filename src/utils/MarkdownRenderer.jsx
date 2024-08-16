import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkFrontmatter from "remark-frontmatter";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import "./MarkdownRenderer.css";

const MarkdownRenderer = () => {
  const { filename } = useParams();
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(true);
  const src = filename ? `${filename}` : '';
  
  useEffect(() => {
    const fetchMarkdown = async () => {
      try {
        const response = await fetch(src);
        if (response.ok) {
          const text = await response.text();
          setMarkdown(text);
        } else {
          console.error("Failed to fetch markdown file:", response.statusText);
        }
      } catch (error) {
        console.error("Error fetching markdown file:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchMarkdown();
  }, [src]);

  if (loading) return <p>Loading...</p>;

  return (
    <div className="markdown-container">
      <ReactMarkdown 
        remarkPlugins={[remarkFrontmatter, remarkMath]} 
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        >
        {markdown}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
