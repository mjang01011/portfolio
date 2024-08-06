import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import path from "path";

const BlogPost = () => {
  const { id } = useParams();
  const [content, setContent] = useState("");

  useEffect(() => {
    const fetchContent = async () => {
      const filePath = path.join("/src/markdowns", `${id}.md`);
      const response = await fetch(filePath);
      const text = await response.text();
      setContent(text);
    };

    fetchContent();
  }, [id]);

  return (
    <div>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
};

export default BlogPost;
