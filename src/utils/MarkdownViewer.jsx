import MarkdownRenderer from "./MarkdownRenderer"
// import { useParams } from "react-router-dom"
const MarkdownViewer = () => {
    // const filename = useParams()
    return (
        <MarkdownRenderer fileName={"RLHF.md"}></MarkdownRenderer>
    )
}

export default MarkdownViewer